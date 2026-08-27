"""
Manual Validation and Annotation Sampling Module
Generates stratified cohorts for human annotation and computes precision/recall bounds.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_stratified_validation_sample(
    studies_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    sample_size: int = 150,
    output_path: str = "data/manual_validation/sampled_validation_cohort.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Creates a balanced sample across clusters, formal/informal capture modes,
    and historical eras for manual verification.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    strata_col = "cluster_label" if "cluster_label" in studies_df.columns else "has_formal_capture"
    
    # Stratified sampling
    sample_df = (
        studies_df.groupby(strata_col, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max(1, sample_size // max(1, studies_df[strata_col].nunique()))), random_state=random_state))
    )

    if len(sample_df) < sample_size:
        remaining_n = sample_size - len(sample_df)
        remaining_pool = studies_df[~studies_df["nct_id"].isin(sample_df["nct_id"])]
        if len(remaining_pool) > 0:
            extra = remaining_pool.sample(min(len(remaining_pool), remaining_n), random_state=random_state)
            sample_df = pd.concat([sample_df, extra], ignore_index=True)

    # Attach match context for easier human review
    top_matches = (
        matches_df.groupby("nct_id")
        .apply(lambda g: "; ".join([f"[{m.field_name}] {m.matched_text}" for m in g.itertuples()][:5]))
        .reset_index(name="matched_snippets")
    )
    
    sample_df = sample_df.merge(top_matches, on="nct_id", how="left")

    # Add empty manual annotation columns
    sample_df["manual_true_tgd_study"] = ""  # 1 = True, 0 = False
    sample_df["manual_tgd_focus_type"] = ""  # e.g., HIV_MSM, GAH_Medical, General_Trial, Psychosocial
    sample_df["manual_notes"] = ""

    export_cols = [
        "nct_id",
        "analysis_year",
        "brief_title",
        "study_type",
        "lead_sponsor_name",
        "has_formal_capture",
        "informal_only",
        "cluster_label" if "cluster_label" in sample_df else "has_formal_capture",
        "matched_snippets",
        "brief_summary",
        "eligibility_criteria_text",
        "manual_true_tgd_study",
        "manual_tgd_focus_type",
        "manual_notes",
    ]
    export_cols = [c for c in export_cols if c in sample_df.columns]
    export_df = sample_df[export_cols]

    export_df.to_csv(out_file, index=False, encoding="utf-8")
    logger.info(f"Exported validation sample ({len(export_df)} records) to {out_file}")
    return export_df


def evaluate_manual_annotations(annotated_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes precision of automated retrieval and clustering against manual annotations.
    """
    valid = annotated_df[annotated_df["manual_true_tgd_study"].isin([0, 1, "0", "1"])].copy()
    if len(valid) == 0:
        return {"error": "No completed manual annotations found."}

    valid["manual_true_tgd_study"] = valid["manual_true_tgd_study"].astype(int)
    precision = float(valid["manual_true_tgd_study"].mean())

    return {
        "annotated_count": len(valid),
        "true_positives": int(valid["manual_true_tgd_study"].sum()),
        "false_positives": int((valid["manual_true_tgd_study"] == 0).sum()),
        "estimated_precision": round(precision, 4),
    }

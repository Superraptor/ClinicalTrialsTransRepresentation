"""
Temporal Analysis and Historical Emergence Module
Quantifies the emergence, growth rate, vocabulary shift, and cluster trajectories over time.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_annual_trajectories(
    df: pd.DataFrame,
    baseline_counts: Optional[Dict[int, int]] = None,
    start_year: int = 2000,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Computes annual trial counts, normalization rates, formal vs informal proportions,
    and cluster distributions across years.
    """
    valid_df = df[df["analysis_year"].notna()].copy()
    valid_df["analysis_year"] = valid_df["analysis_year"].astype(int)
    valid_df = valid_df[(valid_df["analysis_year"] >= start_year) & (valid_df["analysis_year"] <= end_year)]

    years = list(range(start_year, end_year + 1))
    records = []

    for yr in years:
        sub = valid_df[valid_df["analysis_year"] == yr]
        tgd_count = len(sub)
        baseline = baseline_counts.get(yr, 0) if baseline_counts else 0
        rate_per_1000 = (tgd_count / baseline * 1000.0) if baseline > 0 else 0.0

        formal_count = int((sub["has_formal_capture"] == True).sum()) if "has_formal_capture" in sub else 0
        informal_only_count = int((sub["informal_only"] == True).sum()) if "informal_only" in sub else 0
        dual_count = int((sub["dual_formal_informal"] == True).sum()) if "dual_formal_informal" in sub else 0
        
        formal_prop = (formal_count / tgd_count) if tgd_count > 0 else 0.0
        informal_only_prop = (informal_only_count / tgd_count) if tgd_count > 0 else 0.0

        records.append({
            "year": yr,
            "tgd_trial_count": tgd_count,
            "baseline_total_trials": baseline,
            "tgd_rate_per_1000_trials": round(rate_per_1000, 4),
            "formal_capture_count": formal_count,
            "informal_only_count": informal_only_count,
            "dual_capture_count": dual_count,
            "formal_capture_proportion": round(formal_prop, 4),
            "informal_only_proportion": round(informal_only_prop, 4),
        })

    traj_df = pd.DataFrame(records)
    traj_df["cumulative_tgd_trials"] = traj_df["tgd_trial_count"].cumsum()
    return traj_df


def compute_term_trajectories_over_time(
    matches_df: pd.DataFrame,
    studies_df: pd.DataFrame,
    start_year: int = 2000,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Computes the annual frequency and proportion of specific terminology categories
    (e.g., historical psychiatric vs diagnostic vs affirmative identity vs medical affirming).
    """
    merged = matches_df.merge(studies_df[["nct_id", "analysis_year"]], on="nct_id", how="left")
    merged = merged[merged["analysis_year"].notna()].copy()
    merged["analysis_year"] = merged["analysis_year"].astype(int)
    merged = merged[(merged["analysis_year"] >= start_year) & (merged["analysis_year"] <= end_year)]

    # Group by year and category
    term_counts = merged.groupby(["analysis_year", "category"])["nct_id"].nunique().reset_index()
    term_counts.rename(columns={"nct_id": "study_count"}, inplace=True)

    # Pivot to wide format
    pivot_df = term_counts.pivot(index="analysis_year", columns="category", values="study_count").fillna(0)
    pivot_df = pivot_df.reindex(range(start_year, end_year + 1), fill_value=0)
    pivot_df.reset_index(inplace=True)
    pivot_df.columns.name = None
    pivot_df.rename(columns={"index": "year", "analysis_year": "year"}, inplace=True)
    if "year" not in pivot_df.columns:
        pivot_df.rename(columns={pivot_df.columns[0]: "year"}, inplace=True)

    return pivot_df


def compute_cluster_trajectories_over_time(
    df: pd.DataFrame,
    start_year: int = 2000,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Tracks annual counts and relative proportions for each unsupervised cluster.
    """
    valid_df = df[df["analysis_year"].notna()].copy()
    valid_df["analysis_year"] = valid_df["analysis_year"].astype(int)
    valid_df = valid_df[(valid_df["analysis_year"] >= start_year) & (valid_df["analysis_year"] <= end_year)]

    cluster_col = "cluster_label" if "cluster_label" in valid_df.columns else "cluster_id"
    grouped = valid_df.groupby(["analysis_year", cluster_col]).size().reset_index(name="count")

    pivot = grouped.pivot(index="analysis_year", columns=cluster_col, values="count").fillna(0)
    pivot = pivot.reindex(range(start_year, end_year + 1), fill_value=0)
    pivot.reset_index(inplace=True)
    pivot.columns.name = None
    pivot.rename(columns={"index": "year", "analysis_year": "year"}, inplace=True)
    if "year" not in pivot.columns:
        pivot.rename(columns={pivot.columns[0]: "year"}, inplace=True)

    return pivot

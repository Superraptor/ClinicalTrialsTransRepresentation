"""
Quantitative Metrics and Statistical Summary Module
Computes growth rates, odds ratios, funding disparities, and representation metrics.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_growth_rate(years: List[int], counts: List[int]) -> Dict[str, float]:
    """
    Computes Compound Annual Growth Rate (CAGR) and linear trend slope.
    """
    if len(years) < 2 or sum(counts) == 0:
        return {"cagr": 0.0, "slope": 0.0, "r_squared": 0.0, "p_value": 1.0}

    # Linear regression on year vs count
    res = stats.linregress(years, counts)
    
    # CAGR between first non-zero year and last year
    non_zero_idx = [i for i, c in enumerate(counts) if c > 0]
    if len(non_zero_idx) >= 2:
        first_i = non_zero_idx[0]
        last_i = non_zero_idx[-1]
        n_periods = years[last_i] - years[first_i]
        if n_periods > 0 and counts[first_i] > 0:
            cagr = ((counts[last_i] / counts[first_i]) ** (1.0 / n_periods)) - 1.0
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    return {
        "cagr": round(float(cagr), 4),
        "slope": round(float(res.slope), 4),
        "r_squared": round(float(res.rvalue ** 2), 4),
        "p_value": float(res.pvalue),
    }


def compute_funding_odds_ratios(df: pd.DataFrame, group_col: str = "cluster_label") -> pd.DataFrame:
    """
    Computes odds ratio for NIH funding vs. other funding sources across clusters/categories.
    """
    results = []
    total_nih = (df["has_nih_funding"] == True).sum()
    total_non_nih = len(df) - total_nih

    for grp in df[group_col].dropna().unique():
        sub = df[df[group_col] == grp]
        grp_nih = (sub["has_nih_funding"] == True).sum()
        grp_non_nih = len(sub) - grp_nih

        rest_nih = total_nih - grp_nih
        rest_non_nih = total_non_nih - grp_non_nih

        # 2x2 contingency table: [[grp_nih, grp_non_nih], [rest_nih, rest_non_nih]]
        table = [[grp_nih, grp_non_nih], [rest_nih, rest_non_nih]]
        try:
            odds_ratio, p_val = stats.fisher_exact(table)
        except Exception:
            odds_ratio, p_val = 1.0, 1.0

        results.append({
            "group": grp,
            "sample_size": len(sub),
            "nih_funded_count": grp_nih,
            "nih_funded_pct": round(grp_nih / len(sub) * 100, 2) if len(sub) > 0 else 0.0,
            "odds_ratio": round(float(odds_ratio), 3) if np.isfinite(odds_ratio) else np.nan,
            "p_value": round(float(p_val), 4),
        })

    return pd.DataFrame(results).sort_values("nih_funded_pct", ascending=False)


def generate_overall_summary_metrics(
    df: pd.DataFrame,
    baseline_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Generates high-level quantitative summary stats for manuscript text.
    """
    total_tgd_trials = len(df)
    years = df["analysis_year"].dropna().astype(int)
    min_year = int(years.min()) if len(years) > 0 else 2000
    max_year = int(years.max()) if len(years) > 0 else 2026

    # Formal vs informal
    formal_count = int((df["has_formal_capture"] == True).sum()) if "has_formal_capture" in df else 0
    informal_only_count = int((df["informal_only"] == True).sum()) if "informal_only" in df else 0
    dual_count = int((df["dual_formal_informal"] == True).sum()) if "dual_formal_informal" in df else 0

    # Funding breakdown
    nih_count = int((df["has_nih_funding"] == True).sum()) if "has_nih_funding" in df else 0
    industry_count = int((df["has_industry_funding"] == True).sum()) if "has_industry_funding" in df else 0

    # Results reporting
    results_posted_count = int((df["has_results"] == True).sum()) if "has_results" in df else 0

    # Interventional vs Observational
    interventional_count = int((df["study_type"] == "INTERVENTIONAL").sum()) if "study_type" in df else 0
    observational_count = int((df["study_type"] == "OBSERVATIONAL").sum()) if "study_type" in df else 0

    return {
        "total_tgd_trials": total_tgd_trials,
        "temporal_span": f"{min_year} - {max_year}",
        "formal_capture_count": formal_count,
        "formal_capture_pct": round(formal_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "informal_only_count": informal_only_count,
        "informal_only_pct": round(informal_only_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "dual_capture_count": dual_count,
        "dual_capture_pct": round(dual_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "nih_funded_count": nih_count,
        "nih_funded_pct": round(nih_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "industry_funded_count": industry_count,
        "industry_funded_pct": round(industry_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "results_posted_count": results_posted_count,
        "results_posted_pct": round(results_posted_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "interventional_count": interventional_count,
        "interventional_pct": round(interventional_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
        "observational_count": observational_count,
        "observational_pct": round(observational_count / total_tgd_trials * 100, 2) if total_tgd_trials > 0 else 0,
    }

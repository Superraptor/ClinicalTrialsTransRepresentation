"""
Comparative Denominator, Funding ($ USD), NIH Budget Share, and Publication Yield Analytics
Computes platform share, top individual funders/sponsors, financial allocations,
budget share of total NIH appropriations, DSD/intersex disentanglement, and scientific publication rates.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_platform_share_trajectories(
    studies_df: pd.DataFrame,
    baseline_counts: Dict[int, int],
    start_year: int = 2000,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Computes annual and cumulative comparison of TGD studies vs total ClinicalTrials.gov corpus.
    """
    valid_df = studies_df[studies_df["analysis_year"].notna()].copy()
    valid_df["analysis_year"] = valid_df["analysis_year"].astype(int)

    years = list(range(start_year, end_year + 1))
    rows = []
    
    cum_tgd = 0
    cum_platform = 0

    for yr in years:
        tgd_n = int((valid_df["analysis_year"] == yr).sum())
        platform_n = baseline_counts.get(yr, 0)
        
        cum_tgd += tgd_n
        cum_platform += platform_n

        share_pct = (tgd_n / platform_n * 100.0) if platform_n > 0 else 0.0
        rate_per_10k = (tgd_n / platform_n * 10000.0) if platform_n > 0 else 0.0
        cum_share_pct = (cum_tgd / cum_platform * 100.0) if cum_platform > 0 else 0.0

        rows.append({
            "year": yr,
            "annual_tgd_studies": tgd_n,
            "annual_platform_total_studies": platform_n,
            "annual_share_percentage": round(share_pct, 4),
            "annual_rate_per_10000_trials": round(rate_per_10k, 2),
            "cumulative_tgd_studies": cum_tgd,
            "cumulative_platform_total_studies": cum_platform,
            "cumulative_share_percentage": round(cum_share_pct, 4),
        })

    return pd.DataFrame(rows)


def compute_nih_budget_share_trajectories(
    annual_funding_df: pd.DataFrame,
    nih_budget_yaml_path: str = "config/nih_budget.yaml",
) -> pd.DataFrame:
    """
    Computes the proportion of total NIH enacted appropriations allocated to TGD clinical trials per year.
    """
    budget_dict: Dict[int, float] = {}
    p = Path(nih_budget_yaml_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            budget_dict = cfg.get("nih_annual_enacted_budget_billions", {})

    rows = []
    if len(annual_funding_df) > 0:
        for _, row in annual_funding_df.iterrows():
            yr = int(row["year"])
            tgd_dollars = float(row["total_funding_usd"])
            total_nih_billions = float(budget_dict.get(yr, 0.0))
            total_nih_dollars = total_nih_billions * 1e9

            share_of_nih_pct = (tgd_dollars / total_nih_dollars * 100.0) if total_nih_dollars > 0 else 0.0

            rows.append({
                "fiscal_year": yr,
                "tgd_grant_funding_usd": tgd_dollars,
                "tgd_grant_funding_millions": round(tgd_dollars / 1e6, 3),
                "total_nih_enacted_budget_billions": total_nih_billions,
                "tgd_share_of_total_nih_budget_pct": round(share_of_nih_pct, 6),
                "tgd_per_million_nih_dollars": round(share_of_nih_pct * 10000, 4),
            })

    return pd.DataFrame(rows).sort_values("fiscal_year").reset_index(drop=True)


def analyze_cohort_specificity(
    studies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarizes the population specificity: TGD core, SGM broad, DSD/intersex, and endocrine other.
    """
    if "cohort_focus_type" not in studies_df.columns:
        return pd.DataFrame()

    counts = (
        studies_df.groupby("cohort_focus_type")
        .agg(
            study_count=("nct_id", "count"),
            interventional_count=("study_type", lambda s: (s == "INTERVENTIONAL").sum()),
            observational_count=("study_type", lambda s: (s == "OBSERVATIONAL").sum()),
            nih_funded_count=("has_linked_nih_grant" if "has_linked_nih_grant" in studies_df else "lead_sponsor_class", lambda x: (x == True).sum() if "has_linked_nih_grant" in studies_df else 0),
        )
        .reset_index()
    )
    counts["proportion_pct"] = (counts["study_count"] / len(studies_df) * 100.0).round(2)
    return counts.sort_values("study_count", ascending=False).reset_index(drop=True)


def analyze_individual_funders_and_sponsors(
    studies_df: pd.DataFrame,
    top_n: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes top individual sponsors/funders overall and their longitudinal trajectory.
    """
    sponsor_counts = (
        studies_df.groupby(["lead_sponsor_name", "lead_sponsor_class"])
        .agg(
            total_studies=("nct_id", "count"),
            interventional_studies=("study_type", lambda s: (s == "INTERVENTIONAL").sum()),
            observational_studies=("study_type", lambda s: (s == "OBSERVATIONAL").sum()),
            first_year=("analysis_year", "min"),
            latest_year=("analysis_year", "max"),
        )
        .reset_index()
    )
    
    total_trials = len(studies_df)
    sponsor_counts["cohort_share_pct"] = (sponsor_counts["total_studies"] / total_trials * 100.0).round(2)
    top_sponsors_df = sponsor_counts.sort_values("total_studies", ascending=False).head(top_n).reset_index(drop=True)

    top_names = set(top_sponsors_df["lead_sponsor_name"].head(10))
    valid_df = studies_df[studies_df["analysis_year"].notna() & studies_df["lead_sponsor_name"].isin(top_names)].copy()
    valid_df["analysis_year"] = valid_df["analysis_year"].astype(int)

    sponsor_traj_df = valid_df.groupby(["analysis_year", "lead_sponsor_name"]).size().unstack(fill_value=0).reset_index()
    sponsor_traj_df.rename(columns={"analysis_year": "year"}, inplace=True)

    return top_sponsors_df, sponsor_traj_df


def analyze_funding_dollar_allocations(
    awards_df: pd.DataFrame,
    studies_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes financial allocations ($ USD) per fiscal year, per cluster, and per administering IC.
    """
    if len(awards_df) == 0:
        return {
            "total_grant_funding_usd": 0.0,
            "annual_funding_df": pd.DataFrame(),
            "ic_funding_df": pd.DataFrame(),
            "cluster_funding_df": pd.DataFrame(),
        }

    merged = awards_df.merge(
        studies_df[["nct_id", "cluster_label" if "cluster_label" in studies_df else "analysis_year"]],
        on="nct_id",
        how="left",
    )

    total_dollars = float(awards_df["award_amount_usd"].sum())

    annual_funding = (
        awards_df.groupby("fiscal_year")["award_amount_usd"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "total_funding_usd", "count": "award_count", "fiscal_year": "year"})
    )
    annual_funding["total_funding_millions"] = (annual_funding["total_funding_usd"] / 1e6).round(3)

    ic_funding = (
        awards_df[awards_df["administering_ic"] != ""]
        .groupby("administering_ic")["award_amount_usd"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "total_funding_usd", "count": "award_count"})
        .sort_values("total_funding_usd", ascending=False)
    )
    ic_funding["total_funding_millions"] = (ic_funding["total_funding_usd"] / 1e6).round(3)

    cluster_col = "cluster_label" if "cluster_label" in merged.columns else "nct_id"
    cluster_funding = (
        merged.groupby(cluster_col)["award_amount_usd"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "total_funding_usd", "count": "award_count"})
        .sort_values("total_funding_usd", ascending=False)
    )
    cluster_funding["total_funding_millions"] = (cluster_funding["total_funding_usd"] / 1e6).round(3)

    return {
        "total_grant_funding_usd": total_dollars,
        "total_grant_funding_millions": round(total_dollars / 1e6, 2),
        "annual_funding_df": annual_funding,
        "ic_funding_df": ic_funding,
        "cluster_funding_df": cluster_funding,
    }


def analyze_publication_yield(
    pub_df: pd.DataFrame,
    studies_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes scientific publication metrics (PMIDs, PMCIDs, DOIs, open-access rate, top journals).
    """
    if len(pub_df) == 0:
        return {
            "total_publications": 0,
            "studies_with_publications": 0,
            "publication_rate_pct": 0.0,
            "open_access_rate_pct": 0.0,
            "doi_coverage_pct": 0.0,
            "top_journals_df": pd.DataFrame(),
            "cluster_pub_df": pd.DataFrame(),
        }

    total_studies = len(studies_df)
    unique_pmids = pub_df[pub_df["pmid"] != ""]["pmid"].nunique()
    studies_with_pubs = pub_df["nct_id"].nunique()
    
    pmc_count = (pub_df["pmcid"] != "").sum()
    doi_count = (pub_df["doi"] != "").sum()

    pub_rate = (studies_with_pubs / total_studies * 100.0) if total_studies > 0 else 0.0
    oa_rate = (pmc_count / len(pub_df) * 100.0) if len(pub_df) > 0 else 0.0
    doi_rate = (doi_count / len(pub_df) * 100.0) if len(pub_df) > 0 else 0.0

    top_journals = (
        pub_df[pub_df["resolved_journal"] != ""]
        .groupby("resolved_journal")
        .size()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
        .head(15)
    )

    merged = pub_df.merge(
        studies_df[["nct_id", "cluster_label" if "cluster_label" in studies_df else "analysis_year"]],
        on="nct_id",
        how="left",
    )
    cluster_col = "cluster_label" if "cluster_label" in merged.columns else "nct_id"
    cluster_pub_summary = (
        merged.groupby(cluster_col)
        .agg(
            total_citations=("citation", "count"),
            unique_pmids=("pmid", lambda p: (p != "").sum()),
            open_access_articles=("is_open_access", lambda o: (o == "True").sum()),
            unique_trials=("nct_id", "nunique"),
        )
        .reset_index()
    )

    return {
        "total_reference_records": len(pub_df),
        "unique_pmids": unique_pmids,
        "studies_with_publications": studies_with_pubs,
        "publication_rate_pct": round(pub_rate, 2),
        "open_access_rate_pct": round(oa_rate, 2),
        "doi_coverage_pct": round(doi_rate, 2),
        "top_journals_df": top_journals,
        "cluster_pub_df": cluster_pub_summary,
    }

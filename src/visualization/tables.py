"""
Manuscript Summary Table Generator
Formats structured tables (Markdown, CSV, LaTeX) for inclusion in the manuscript.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def generate_table1_cohort_summary(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 1: Cohort characteristics, study design, funding, and formal vs informal capture.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table1_cohort_summary.csv"
    out_md = out_path / "table1_cohort_summary.md"

    total = len(df)
    
    rows = [
        {"Characteristic": "Total TGD-Related Studies (N)", "Count / Value": f"{total:,}", "Percentage (%)": "100.0%"},
        {"Characteristic": "Study First Posted Span", "Count / Value": metrics.get("temporal_span", ""), "Percentage (%)": "-"},
        
        # Capture Mechanism
        {"Characteristic": "--- Capture Mechanism ---", "Count / Value": "", "Percentage (%)": ""},
        {"Characteristic": "Formal Metadata Capture (MeSH / genderBased / Keywords)", "Count / Value": f"{metrics.get('formal_capture_count', 0):,}", "Percentage (%)": f"{metrics.get('formal_capture_pct', 0)}%"},
        {"Characteristic": "Informal-Only Capture (Hidden in Free-Text)", "Count / Value": f"{metrics.get('informal_only_count', 0):,}", "Percentage (%)": f"{metrics.get('informal_only_pct', 0)}%"},
        {"Characteristic": "Dual Formal & Informal Capture", "Count / Value": f"{metrics.get('dual_capture_count', 0):,}", "Percentage (%)": f"{metrics.get('dual_capture_pct', 0)}%"},
        
        # Funding
        {"Characteristic": "--- Funding & Sponsorship ---", "Count / Value": "", "Percentage (%)": ""},
        {"Characteristic": "NIH / Federal Funding", "Count / Value": f"{metrics.get('nih_funded_count', 0):,}", "Percentage (%)": f"{metrics.get('nih_funded_pct', 0)}%"},
        {"Characteristic": "Industry Sponsored / Supported", "Count / Value": f"{metrics.get('industry_funded_count', 0):,}", "Percentage (%)": f"{metrics.get('industry_funded_pct', 0)}%"},
        
        # Study Type
        {"Characteristic": "--- Study Type ---", "Count / Value": "", "Percentage (%)": ""},
        {"Characteristic": "Interventional Clinical Trials", "Count / Value": f"{metrics.get('interventional_count', 0):,}", "Percentage (%)": f"{metrics.get('interventional_pct', 0)}%"},
        {"Characteristic": "Observational Studies", "Count / Value": f"{metrics.get('observational_count', 0):,}", "Percentage (%)": f"{metrics.get('observational_pct', 0)}%"},
        
        # Results
        {"Characteristic": "--- Results Reporting ---", "Count / Value": "", "Percentage (%)": ""},
        {"Characteristic": "Studies with Posted Results", "Count / Value": f"{metrics.get('results_posted_count', 0):,}", "Percentage (%)": f"{metrics.get('results_posted_pct', 0)}%"},
    ]

    t1_df = pd.DataFrame(rows)
    t1_df.to_csv(out_file, index=False)
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 1: ClinicalTrials.gov TGD Cohort Characteristics\n\n")
        f.write(t1_df.to_markdown(index=False))

    logger.info(f"Generated Table 1 at {out_file} and {out_md}")
    return str(out_file)


def generate_table2_cluster_characterization(
    cluster_summaries: Dict[int, Dict[str, Any]],
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 2: Unsupervised cluster profiles, top distinctive keywords, temporal profile, and exemplar trials.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table2_unsupervised_clusters.csv"
    out_md = out_path / "table2_unsupervised_clusters.md"

    rows = []
    for c_id, info in cluster_summaries.items():
        top_kws = ", ".join(info.get("top_keywords", [])[:5])
        temp = info.get("temporal_profile", {})
        fund = info.get("funding_profile", {})
        reps = info.get("representative_studies", [])
        exemplar_nct = reps[0]["nct_id"] if reps else "-"

        rows.append({
            "Cluster ID": f"C{c_id}",
            "Descriptive Cluster Label": info.get("cluster_label", ""),
            "Study Count (N)": f"{info.get('study_count', 0):,}",
            "Share (%)": f"{info.get('proportion_pct', 0.0)}%",
            "Top c-TF-IDF Distinctive Terms": top_kws,
            "Mean Year": temp.get("mean_year", "-"),
            "NIH Funding (%)": f"{fund.get('nih_funding_pct', 0)}%",
            "Formal Capture (%)": f"{info.get('formal_capture_pct', 0)}%",
            "Exemplar NCT ID": exemplar_nct,
        })

    t2_df = pd.DataFrame(rows)
    t2_df.to_csv(out_file, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 2: Unsupervised Thematic Cluster Profiles and Characterization\n\n")
        f.write(t2_df.to_markdown(index=False))

    logger.info(f"Generated Table 2 at {out_file} and {out_md}")
    return str(out_file)


def generate_table3_concordance_matrix(
    concordance_df: pd.DataFrame,
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 3: Structural discordance between formal fields and free text.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table3_structural_discordance.csv"
    out_md = out_path / "table3_structural_discordance.md"

    flags = concordance_df["discordance_flags"].str.split(", ").explode()
    flag_counts = flags.value_counts().reset_index()
    flag_counts.columns = ["Discordance Pattern / State", "Count (N)"]
    flag_counts["Percentage of Cohort (%)"] = (flag_counts["Count (N)"] / len(concordance_df) * 100).round(2).astype(str) + "%"

    flag_counts.to_csv(out_file, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 3: Structural Discordance between Formal Eligibility Fields and Textual Criteria\n\n")
        f.write(flag_counts.to_markdown(index=False))

    logger.info(f"Generated Table 3 at {out_file} and {out_md}")
    return str(out_file)


def generate_table4_publications_summary(
    pub_metrics: Dict[str, Any],
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 4: Scientific publication output, PMIDs, PMCIDs, DOIs, and journal distribution.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table4_publications_summary.csv"
    out_md = out_path / "table4_publications_summary.md"

    rows = [
        {"Publication Metric": "Total Reference / Publication Records", "Value": f"{pub_metrics.get('total_reference_records', 0):,}"},
        {"Publication Metric": "Unique Resolved PMIDs", "Value": f"{pub_metrics.get('unique_pmids', 0):,}"},
        {"Publication Metric": "Studies with Indexed Publications", "Value": f"{pub_metrics.get('studies_with_publications', 0):,} ({pub_metrics.get('publication_rate_pct', 0)}%)"},
        {"Publication Metric": "Open Access Coverage (PMCID Available)", "Value": f"{pub_metrics.get('open_access_rate_pct', 0)}%"},
        {"Publication Metric": "DOI Indexing Coverage", "Value": f"{pub_metrics.get('doi_coverage_pct', 0)}%"},
    ]
    t4_df = pd.DataFrame(rows)
    t4_df.to_csv(out_file, index=False)

    top_j = pub_metrics.get("top_journals_df", pd.DataFrame())
    cluster_p = pub_metrics.get("cluster_pub_df", pd.DataFrame())

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 4: Associated Publications, PMIDs, PMCIDs, and DOI Indexing\n\n")
        f.write("### Overall Publication Output\n\n")
        f.write(t4_df.to_markdown(index=False))
        if len(top_j) > 0:
            f.write("\n\n### Top Publishing Journals\n\n")
            f.write(top_j.to_markdown(index=False))
        if len(cluster_p) > 0:
            f.write("\n\n### Publication Yield by Unsupervised Thematic Cluster\n\n")
            f.write(cluster_p.to_markdown(index=False))

    logger.info(f"Generated Table 4 at {out_file} and {out_md}")
    return str(out_file)


def generate_table5_funding_and_sponsors(
    top_sponsors_df: pd.DataFrame,
    funding_res: Dict[str, Any],
    budget_share_df: Optional[pd.DataFrame] = None,
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 5: Top 20 individual funders & sponsors, NIH RePORTER funding amounts ($ USD), and share of total NIH enacted appropriations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table5_funding_and_sponsors.csv"
    out_md = out_path / "table5_funding_and_sponsors.md"

    top_sponsors_df.to_csv(out_file, index=False)

    ic_df = funding_res.get("ic_funding_df", pd.DataFrame())
    annual_f = funding_res.get("annual_funding_df", pd.DataFrame())

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 5: Institutional Funders, Lead Sponsors, and Funding Capital ($ USD)\n\n")
        f.write(f"**Total Documented NIH Grant Funding**: ${funding_res.get('total_grant_funding_millions', 0):,.2f} Million USD\n\n")
        f.write("### Top 20 Individual Lead Sponsors and Funders\n\n")
        f.write(top_sponsors_df.to_markdown(index=False))
        if len(ic_df) > 0:
            f.write("\n\n### NIH Institute & Center (IC) Funding Breakdown\n\n")
            f.write(ic_df.to_markdown(index=False))
        if budget_share_df is not None and len(budget_share_df) > 0:
            f.write("\n\n### TGD Clinical Trial Grant Share of Total Enacted NIH Appropriations (2009–2026)\n\n")
            f.write(budget_share_df.to_markdown(index=False))
        elif len(annual_f) > 0:
            f.write("\n\n### Annual Fiscal Funding Allocations\n\n")
            f.write(annual_f.to_markdown(index=False))

    logger.info(f"Generated Table 5 at {out_file} and {out_md}")
    return str(out_file)


def generate_table6_platform_share_comparison(
    platform_df: pd.DataFrame,
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 6: Longitudinal comparison of TGD studies vs total ClinicalTrials.gov studies per year.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table6_platform_share_comparison.csv"
    out_md = out_path / "table6_platform_share_comparison.md"

    platform_df.to_csv(out_file, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 6: Annual and Cumulative Share of TGD Research in ClinicalTrials.gov (2000–2026)\n\n")
        f.write(platform_df.to_markdown(index=False))

    logger.info(f"Generated Table 6 at {out_file} and {out_md}")
    return str(out_file)


def generate_table7_dsd_intersex_disentanglement(
    cohort_counts_df: pd.DataFrame,
    output_dir: str = "reports/tables",
) -> str:
    """
    Table 7: Disentanglement of TGD Core vs DSD/Intersex and SGM Broad Umbrella Populations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "table7_dsd_intersex_disentanglement.csv"
    out_md = out_path / "table7_dsd_intersex_disentanglement.md"

    if len(cohort_counts_df) > 0:
        cohort_counts_df.to_csv(out_file, index=False)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Table 7: Population Specificity & DSD/Intersex Disentanglement\n\n")
            f.write(cohort_counts_df.to_markdown(index=False))

    logger.info(f"Generated Table 7 at {out_file} and {out_md}")
    return str(out_file)

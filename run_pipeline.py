"""
End-to-End Orchestrator CLI for ClinicalTrials.gov Transgender & Gender-Diverse Pipeline

Usage:
    python run_pipeline.py --mode all          # Run end-to-end extraction, parsing, clustering, publications, funding, and evaluation
    python run_pipeline.py --mode demo         # Run fast pipeline demo using a representative sample
    python run_pipeline.py --mode ingest       # Run API v2 query sweep ingestion
    python run_pipeline.py --mode baseline     # Collect annual denominator totals (1999-2026)
    python run_pipeline.py --mode parse        # Parse cached JSONs into structured datasets
    python run_pipeline.py --mode cluster      # Run unsupervised clustering & characterization
    python run_pipeline.py --mode publications # Extract & resolve PMIDs, PMCIDs, DOIs
    python run_pipeline.py --mode funding      # Retrieve NIH RePORTER grant dollar amounts
    python run_pipeline.py --mode evaluate     # Run metrics, temporal analysis, and tables
    python run_pipeline.py --mode visualize    # Generate publication figures
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml

from src.ingestion.api_client import ClinicalTrialsApiClient
from src.ingestion.baseline_counts import fetch_annual_baseline_counts
from src.ingestion.publication_extractor import PublicationExtractor
from src.ingestion.citation_crosslink import CitationCrossLinker
from src.ingestion.funding_reporter import FundingReporterClient
from src.parsing.trial_parser import parse_study_record
from src.parsing.eligibility_parser import parse_eligibility_rules
from src.parsing.term_extractor import TermExtractor
from src.classification.unsupervised_clustering import TrialClusteringEngine
from src.classification.taxonomy import BenchmarkTaxonomyClassifier
from src.classification.dsd_intersex_classifier import CohortSpecificityClassifier
from src.classification.concordance import compute_concordance_matrix
from src.evaluation.temporal_analysis import (
    compute_annual_trajectories,
    compute_cluster_trajectories_over_time,
    compute_term_trajectories_over_time,
)
from src.evaluation.comparative_analytics import (
    analyze_cohort_specificity,
    analyze_funding_dollar_allocations,
    analyze_individual_funders_and_sponsors,
    analyze_publication_yield,
    compute_nih_budget_share_trajectories,
    compute_platform_share_trajectories,
)
from src.evaluation.metrics import (
    compute_funding_odds_ratios,
    compute_growth_rate,
    generate_overall_summary_metrics,
)
from src.evaluation.validation import create_stratified_validation_sample
from src.visualization.figures import (
    plot_cluster_trajectories,
    plot_formal_vs_informal_capture,
    plot_funding_dollars_and_sponsors,
    plot_historical_emergence,
    plot_publication_yield_and_open_access,
    plot_terminology_trajectories,
    plot_unsupervised_clusters_umap,
)
from src.visualization.network_map import (
    build_investigator_trial_publication_graph,
    plot_interactive_citation_network,
    plot_static_citation_network,
)
from src.visualization.tables import (
    generate_table1_cohort_summary,
    generate_table2_cluster_characterization,
    generate_table3_concordance_matrix,
    generate_table4_publications_summary,
    generate_table5_funding_and_sponsors,
    generate_table6_platform_share_comparison,
    generate_table7_dsd_intersex_disentanglement,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lexicon(lexicon_path: str = "config/lexicon.yaml") -> Dict[str, Any]:
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mesh_config(mesh_path: str = "config/mesh_terms.yaml") -> Dict[str, Any]:
    with open(mesh_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_collection(cfg: Dict[str, Any]) -> Dict[int, int]:
    logger.info("--- Step: Collecting Baseline Denominator Counts ---")
    start_yr = cfg.get("analysis", {}).get("start_year", 1999)
    end_yr = cfg.get("analysis", {}).get("end_year", 2026)
    out_file = cfg.get("paths", {}).get("data_raw_baseline", "data/raw/baseline_counts.json")
    return fetch_annual_baseline_counts(start_year=start_yr, end_year=end_yr, output_path=out_file)


def run_ingestion(cfg: Dict[str, Any], lexicon: Dict[str, Any], mesh_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("--- Step: ClinicalTrials.gov API Ingestion ---")
    api_cfg = cfg.get("api", {})
    client = ClinicalTrialsApiClient(
        base_url=api_cfg.get("base_url", "https://clinicaltrials.gov/api/v2/studies"),
        page_size=api_cfg.get("page_size", 100),
        rate_limit_delay_sec=api_cfg.get("rate_limit_delay_sec", 0.25),
        max_retries=api_cfg.get("max_retries", 5),
        cache_dir=cfg.get("paths", {}).get("data_raw_api", "data/raw/api_responses"),
    )

    queries = lexicon.get("api_search_queries", ["transgender", "gender dysphoria"])
    mesh_terms = [m["descriptor_name"] for m in mesh_cfg.get("mesh_descriptors", [])]

    consolidated_path = "data/raw/consolidated_studies.json"
    dedup_dict = client.query_sweep(queries=queries, mesh_terms=mesh_terms, save_consolidated_path=consolidated_path)
    return list(dedup_dict.values())


def run_parsing_and_extraction(
    raw_studies: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("--- Step: Parsing Raw Studies & Extracting TGD Concepts ---")
    
    parsed_records = []
    eligibility_records = []
    
    for st in raw_studies:
        rec = parse_study_record(st)
        parsed_records.append(rec)
        
        rules = parse_eligibility_rules(rec["nct_id"], rec.get("eligibility_criteria_text", ""))
        eligibility_records.extend(rules)

    studies_df = pd.DataFrame(parsed_records)
    rules_df = pd.DataFrame(eligibility_records)

    logger.info(f"Parsed {len(studies_df)} studies and {len(rules_df)} eligibility rules.")

    # Run Term Extractor
    extractor = TermExtractor()
    evaluated_meta = []
    all_match_rows = []

    for _, row in studies_df.iterrows():
        eval_res = extractor.evaluate_study_record(row.to_dict())
        matches = eval_res.pop("matches", [])
        evaluated_meta.append(eval_res)
        all_match_rows.extend(matches)

    eval_df = pd.DataFrame(evaluated_meta)
    matches_df = pd.DataFrame(all_match_rows)

    studies_df = studies_df.merge(eval_df, on="nct_id", how="left")

    # Run DSD / Intersex and Cohort Specificity Classifier
    dsd_clf = CohortSpecificityClassifier()
    dsd_df = dsd_clf.classify_dataframe(studies_df)
    studies_df = studies_df.merge(dsd_df[["nct_id", "cohort_focus_type", "is_tgd_core", "is_dsd_intersex", "is_sgm_broad"]], on="nct_id", how="left")

    # Filter to candidate cohort
    tgd_studies_df = studies_df[
        (studies_df["total_match_count"] > 0) | (studies_df["has_formal_capture"] == True)
    ].copy().reset_index(drop=True)

    logger.info(f"Confirmed {len(tgd_studies_df)} candidate studies. Core TGD: {(tgd_studies_df['is_tgd_core'] == True).sum()}, DSD/Intersex specific: {(tgd_studies_df['cohort_focus_type'] == 'DSD_INTERSEX_CONGENITAL').sum()}")

    for col in tgd_studies_df.select_dtypes(include=["object"]).columns:
        tgd_studies_df[col] = tgd_studies_df[col].fillna("").astype(str)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    tgd_studies_df.to_parquet(out_dir / "tgd_studies.parquet", index=False)
    tgd_studies_df.to_csv(out_dir / "tgd_studies.csv", index=False)
    rules_df.to_parquet(out_dir / "eligibility_rules.parquet", index=False)
    matches_df.to_parquet(out_dir / "lexicon_matches.parquet", index=False)

    return tgd_studies_df, rules_df, matches_df


def run_unsupervised_clustering(
    studies_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    logger.info("--- Step: Unsupervised Clustering & Post-Hoc Characterization ---")
    clust_cfg = cfg.get("clustering", {})
    
    engine = TrialClusteringEngine(
        n_clusters=clust_cfg.get("n_clusters", 8),
        min_cluster_size=clust_cfg.get("min_cluster_size", 15),
        use_embeddings=True,
        embedding_model_name=clust_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        random_state=clust_cfg.get("random_state", 42),
    )

    clustered_df, summaries = engine.fit_transform(studies_df)

    out_dir = Path("data/processed")
    for col in clustered_df.select_dtypes(include=["object"]).columns:
        clustered_df[col] = clustered_df[col].fillna("").astype(str)

    clustered_df.to_parquet(out_dir / "unsupervised_clusters.parquet", index=False)
    clustered_df.to_csv(out_dir / "unsupervised_clusters.csv", index=False)

    with open(out_dir / "cluster_characterization.json", "w", encoding="utf-8") as f:
        cleaned_summaries = {}
        for c_id, info in summaries.items():
            cleaned_summaries[str(c_id)] = {
                "cluster_id": int(c_id),
                "cluster_label": info["cluster_label"],
                "study_count": int(info["study_count"]),
                "proportion_pct": float(info["proportion_pct"]),
                "top_keywords": info["top_keywords"],
                "top_c_tf_idf_terms": [[t[0], float(t[1])] for t in info["top_c_tf_idf_terms"]],
                "representative_studies": info["representative_studies"],
                "temporal_profile": info["temporal_profile"],
                "funding_profile": info["funding_profile"],
                "formal_capture_pct": float(info["formal_capture_pct"]),
            }
        json.dump(cleaned_summaries, f, indent=2)

    logger.info(f"Completed clustering. Generated {len(summaries)} characterized thematic clusters.")
    return clustered_df, summaries


def run_publications_and_funding(
    raw_studies: List[Dict[str, Any]],
    studies_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    logger.info("--- Step: Extracting Publications, Crossref Cross-Citations & NIH RePORTER Funding ($) ---")
    
    cohort_nct_ids = set(studies_df["nct_id"].unique())
    pub_ext = PublicationExtractor()
    pub_df = pub_ext.process_and_enrich_studies(raw_studies, cohort_nct_ids=cohort_nct_ids)
    pub_metrics = analyze_publication_yield(pub_df, studies_df)

    # Direct Citation Cross-Linkages
    crosslinker = CitationCrossLinker()
    crosslinks_df = crosslinker.discover_internal_cross_citations(pub_df)

    fund_client = FundingReporterClient()
    awards_df, fund_summary_df = fund_client.fetch_cohort_funding(studies_df)
    funding_res = analyze_funding_dollar_allocations(awards_df, studies_df)

    return pub_df, crosslinks_df, awards_df, fund_summary_df, pub_metrics, funding_res


def run_evaluation_and_tables(
    studies_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    cluster_summaries: Dict[int, Dict[str, Any]],
    baseline_counts: Dict[int, int],
    pub_metrics: Dict[str, Any],
    funding_res: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("--- Step: Computing Quantitative Metrics & Generating Tables ---")

    bench_clf = BenchmarkTaxonomyClassifier()
    bench_df = bench_clf.classify_dataframe(studies_df)
    studies_df = studies_df.merge(bench_df, on="nct_id", how="left")

    concordance_df = compute_concordance_matrix(studies_df)
    studies_df = studies_df.merge(concordance_df[["nct_id", "is_concordant", "discordance_flags", "discordance_count"]], on="nct_id", how="left")

    summary_metrics = generate_overall_summary_metrics(studies_df, baseline_counts)
    summary_metrics["total_nih_grant_funding_millions"] = funding_res.get("total_grant_funding_millions", 0.0)
    summary_metrics["unique_publications_count"] = pub_metrics.get("unique_pmids", 0)
    summary_metrics["publication_rate_pct"] = pub_metrics.get("publication_rate_pct", 0.0)

    traj_df = compute_annual_trajectories(studies_df, baseline_counts)
    term_traj_df = compute_term_trajectories_over_time(matches_df, studies_df)
    cluster_traj_df = compute_cluster_trajectories_over_time(studies_df)
    platform_df = compute_platform_share_trajectories(studies_df, baseline_counts)
    top_sponsors_df, sponsor_traj_df = analyze_individual_funders_and_sponsors(studies_df)
    
    annual_funding = funding_res.get("annual_funding_df", pd.DataFrame())
    budget_share_df = compute_nih_budget_share_trajectories(annual_funding)
    cohort_specificity_df = analyze_cohort_specificity(studies_df)

    generate_table1_cohort_summary(studies_df, summary_metrics)
    generate_table2_cluster_characterization(cluster_summaries)
    generate_table3_concordance_matrix(concordance_df)
    generate_table4_publications_summary(pub_metrics)
    generate_table5_funding_and_sponsors(top_sponsors_df, funding_res, budget_share_df)
    generate_table6_platform_share_comparison(platform_df)
    generate_table7_dsd_intersex_disentanglement(cohort_specificity_df)

    create_stratified_validation_sample(studies_df, matches_df, sample_size=100)

    with open("reports/summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("Evaluation and table generation complete.")
    return {
        "summary_metrics": summary_metrics,
        "traj_df": traj_df,
        "term_traj_df": term_traj_df,
        "cluster_traj_df": cluster_traj_df,
        "platform_df": platform_df,
        "top_sponsors_df": top_sponsors_df,
        "sponsor_traj_df": sponsor_traj_df,
        "budget_share_df": budget_share_df,
        "cohort_specificity_df": cohort_specificity_df,
        "studies_df": studies_df,
    }


def run_visualizations(
    studies_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    term_traj_df: pd.DataFrame,
    cluster_traj_df: pd.DataFrame,
    cluster_summaries: Dict[int, Dict[str, Any]],
    funding_res: Dict[str, Any],
    top_sponsors_df: pd.DataFrame,
    pub_metrics: Dict[str, Any],
    awards_df: pd.DataFrame,
    pub_df: pd.DataFrame,
    crosslinks_df: Optional[pd.DataFrame] = None,
):
    logger.info("--- Step: Generating Publication-Grade Figures & Network Map ---")
    plot_historical_emergence(traj_df)
    plot_formal_vs_informal_capture(traj_df)
    plot_terminology_trajectories(term_traj_df)
    plot_unsupervised_clusters_umap(studies_df, cluster_summaries)
    plot_cluster_trajectories(cluster_traj_df)
    plot_funding_dollars_and_sponsors(funding_res, top_sponsors_df)
    plot_publication_yield_and_open_access(pub_metrics)

    # Figure 8: Network Map across studies with direct cross-citations and shared publications
    G = build_investigator_trial_publication_graph(
        studies_df,
        awards_df,
        pub_df,
        crosslinks_df=crosslinks_df,
        include_all_pubs=True,
    )
    plot_static_citation_network(
        G,
        studies_df=studies_df,
        awards_df=awards_df,
        pub_df=pub_df,
        crosslinks_df=crosslinks_df,
    )
    plot_interactive_citation_network(G)
    logger.info("All 8 figures and interactive citation network generated in reports/figures/")


def main():
    parser = argparse.ArgumentParser(description="ClinicalTrials.gov TGD Representation Pipeline")
    parser.add_argument("--mode", choices=["all", "demo", "ingest", "baseline", "parse", "cluster", "publications", "funding", "evaluate", "visualize"], default="all", help="Pipeline execution mode")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Path to pipeline configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lexicon = load_lexicon("config/lexicon.yaml")
    mesh_cfg = load_mesh_config("config/mesh_terms.yaml")

    baseline = run_baseline_collection(cfg)

    if args.mode == "demo":
        logger.info("=== Running Pipeline in Demo / Validation Mode ===")
        demo_queries = ["transgender", "gender dysphoria", "transsexualism", "gender affirming", "cross-sex hormone", "vaginoplasty", "FTM", "MTF"]
        api_cfg = cfg.get("api", {})
        client = ClinicalTrialsApiClient(
            base_url=api_cfg.get("base_url", "https://clinicaltrials.gov/api/v2/studies"),
            page_size=50,
            rate_limit_delay_sec=0.2,
        )
        dedup_studies = client.query_sweep(queries=demo_queries, mesh_terms=["Transgender Persons", "Gender Dysphoria", "Transsexualism"])
        raw_studies = list(dedup_studies.values())

        studies_df, rules_df, matches_df = run_parsing_and_extraction(raw_studies, cfg)
        clustered_df, summaries = run_unsupervised_clustering(studies_df, cfg)
        pub_df, crosslinks_df, awards_df, fund_summary_df, pub_metrics, funding_res = run_publications_and_funding(raw_studies, clustered_df)
        
        eval_res = run_evaluation_and_tables(clustered_df, matches_df, summaries, baseline, pub_metrics, funding_res, cfg)

        run_visualizations(
            eval_res["studies_df"],
            eval_res["traj_df"],
            eval_res["term_traj_df"],
            eval_res["cluster_traj_df"],
            summaries,
            funding_res,
            eval_res["top_sponsors_df"],
            pub_metrics,
            awards_df,
            pub_df,
            crosslinks_df,
        )
        logger.info("Demo execution completed successfully!")
        return

    # Ingestion or cache load
    consolidated = Path("data/raw/consolidated_studies.json")
    if args.mode in ("ingest", "all") or not consolidated.exists():
        raw_studies = run_ingestion(cfg, lexicon, mesh_cfg)
    else:
        with open(consolidated, "r", encoding="utf-8") as f:
            raw_studies = json.load(f)

    studies_df, rules_df, matches_df = run_parsing_and_extraction(raw_studies, cfg)
    clustered_df, summaries = run_unsupervised_clustering(studies_df, cfg)
    pub_df, crosslinks_df, awards_df, fund_summary_df, pub_metrics, funding_res = run_publications_and_funding(raw_studies, clustered_df)
    eval_res = run_evaluation_and_tables(clustered_df, matches_df, summaries, baseline, pub_metrics, funding_res, cfg)

    run_visualizations(
        eval_res["studies_df"],
        eval_res["traj_df"],
        eval_res["term_traj_df"],
        eval_res["cluster_traj_df"],
        summaries,
        funding_res,
        eval_res["top_sponsors_df"],
        pub_metrics,
        awards_df,
        pub_df,
        crosslinks_df,
    )

    logger.info("=== Full Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    main()

"""
Unit tests for publication extraction and funding analytics.
"""

import pytest
from pathlib import Path
import pandas as pd
from src.ingestion.publication_extractor import PublicationExtractor
from src.ingestion.funding_reporter import FundingReporterClient
from src.evaluation.comparative_analytics import compute_platform_share_trajectories, analyze_individual_funders_and_sponsors


def test_publication_extractor_raw():
    extractor = PublicationExtractor()
    mock_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567"},
            "referencesModule": {
                "references": [
                    {
                        "pmid": "29185432",
                        "type": "RESULT",
                        "citation": "Smith J et al. Transgender health outcomes. Lancet. 2018; doi: 10.1016/S0140-6736(18)30001-X."
                    }
                ]
            }
        }
    }
    refs = extractor.extract_raw_references(mock_study)
    assert len(refs) == 1
    assert refs[0]["pmid"] == "29185432"
    assert refs[0]["reference_type"] == "RESULT"
    assert "10.1016/S0140-6736(18)30001-X" in refs[0]["extracted_doi"]


def test_publication_extractor_cohort_filtering():
    extractor = PublicationExtractor()
    mock_studies = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "referencesModule": {"references": [{"pmid": "11111111", "type": "RESULT", "citation": "Study 1 paper"}]}
            }
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002"},
                "referencesModule": {"references": [{"pmid": "22222222", "type": "RESULT", "citation": "Study 2 paper"}]}
            }
        },
    ]
    # Filter to only NCT00000001
    filtered_df = extractor.process_and_enrich_studies(mock_studies, cohort_nct_ids={"NCT00000001"}, output_parquet="tests/test_pubs.parquet")
    assert len(filtered_df) == 1
    assert filtered_df["nct_id"].iloc[0] == "NCT00000001"
    assert filtered_df["pmid"].iloc[0] == "11111111"
    if Path("tests/test_pubs.parquet").exists():
        Path("tests/test_pubs.parquet").unlink()
    if Path("tests/test_pubs.csv").exists():
        Path("tests/test_pubs.csv").unlink()


def test_funding_grant_extraction():
    client = FundingReporterClient()
    mock_study = {
        "nct_id": "NCT09876543",
        "org_study_id": "R01MH112233",
        "brief_summary": "Funded by NIH grant U01AI099887 evaluating PrEP.",
        "detailed_description": "",
    }
    grants = client.extract_grant_numbers_from_study(mock_study)
    assert any("MH112233" in g or "R01MH112233" in g for g in grants)
    assert any("AI099887" in g or "U01AI099887" in g for g in grants)


def test_platform_share_trajectories():
    mock_df = pd.DataFrame([
        {"nct_id": "NCT001", "analysis_year": 2020},
        {"nct_id": "NCT002", "analysis_year": 2020},
        {"nct_id": "NCT003", "analysis_year": 2021},
    ])
    baseline = {2020: 30000, 2021: 35000}
    share_df = compute_platform_share_trajectories(mock_df, baseline, start_year=2020, end_year=2021)
    
    assert len(share_df) == 2
    assert "annual_share_percentage" in share_df.columns
    assert "annual_rate_per_10000_trials" in share_df.columns
    assert share_df.loc[share_df["year"] == 2020, "annual_tgd_studies"].values[0] == 2


def test_dsd_intersex_classifier():
    from src.classification.dsd_intersex_classifier import CohortSpecificityClassifier
    clf = CohortSpecificityClassifier()

    # Core TGD trial
    tgd_study = {"brief_title": "Gender-affirming hormone therapy in transgender youth", "brief_summary": "Studying estradiol in trans women"}
    res1 = clf.classify_study(tgd_study)
    assert res1["cohort_focus_type"] == "TGD_CORE_SPECIFIC"
    assert res1["is_tgd_core"] is True
    assert res1["is_dsd_intersex"] is False

    # Congenital DSD trial
    dsd_study = {"brief_title": "Evaluation of feminizing genitoplasty in children with disorders of sex development", "brief_summary": "Congenital adrenal hyperplasia cohort"}
    res2 = clf.classify_study(dsd_study)
    assert res2["cohort_focus_type"] == "DSD_INTERSEX_CONGENITAL"
    assert res2["is_dsd_intersex"] is True
    assert res2["is_tgd_core"] is False


def test_nih_budget_share_calculation():
    from src.evaluation.comparative_analytics import compute_nih_budget_share_trajectories
    annual_funding = pd.DataFrame([
        {"year": 2019, "total_funding_usd": 15281400.0, "award_count": 20},
        {"year": 2020, "total_funding_usd": 14663700.0, "award_count": 25},
    ])
    share_df = compute_nih_budget_share_trajectories(annual_funding)
    assert len(share_df) == 2
    assert "tgd_share_of_total_nih_budget_pct" in share_df.columns
    assert share_df.loc[share_df["fiscal_year"] == 2019, "total_nih_enacted_budget_billions"].values[0] == 39.314


def test_citation_crosslinker_logic():
    from src.ingestion.citation_crosslink import CitationCrossLinker
    crosslinker = CitationCrossLinker(cache_path="tests/test_crosslinks_cache.json")
    crosslinker.cache = {
        "1001": ["1002", "9999"], # 1002 is in cohort, 9999 is external
        "1002": ["1003"],
        "1003": [],
    }
    dummy_pubs = pd.DataFrame([
        {"pmid": "1001", "nct_id": "NCT001", "resolved_journal": "J1", "resolved_pub_year": 2020},
        {"pmid": "1002", "nct_id": "NCT002", "resolved_journal": "J2", "resolved_pub_year": 2021},
        {"pmid": "1003", "nct_id": "NCT003", "resolved_journal": "J3", "resolved_pub_year": 2022},
    ])
    edges_df = crosslinker.discover_internal_cross_citations(dummy_pubs)
    # Only internal edges retained (1002 citing 1001, 1003 citing 1002)
    assert len(edges_df) == 2
    assert "9999" not in edges_df["citing_pmid"].values
    assert "9999" not in edges_df["cited_pmid"].values
    if Path("tests/test_crosslinks_cache.json").exists():
        Path("tests/test_crosslinks_cache.json").unlink()


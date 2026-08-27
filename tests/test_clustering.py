"""
Unit tests for unsupervised clustering and characterization.
"""

import pandas as pd
import pytest
from src.classification.unsupervised_clustering import TrialClusteringEngine
from src.classification.concordance import evaluate_study_concordance


def test_clustering_and_characterization():
    mock_data = [
        {
            "nct_id": f"NCT000000{i:02d}",
            "brief_title": f"HIV PrEP adherence in transgender women study {i}",
            "brief_summary": "Evaluating pre-exposure prophylaxis and antiretroviral therapy among trans women and MSM.",
            "conditions": "HIV Infections",
            "interventions": "Emtricitabine / Tenofovir",
            "keywords": "HIV, PrEP, Transgender",
            "condition_mesh_terms": "HIV Infections",
            "analysis_year": 2018 + (i % 5),
            "has_nih_funding": True,
            "has_industry_funding": False,
            "has_formal_capture": True,
        }
        for i in range(1, 10)
    ] + [
        {
            "nct_id": f"NCT000000{i:02d}",
            "brief_title": f"Gender-affirming vaginoplasty surgical outcomes {i}",
            "brief_summary": "Surgical techniques in penile inversion vaginoplasty for gender confirmation.",
            "conditions": "Gender Dysphoria",
            "interventions": "Vaginoplasty surgery",
            "keywords": "Gender Confirmation Surgery, Vaginoplasty",
            "condition_mesh_terms": "Gender Dysphoria",
            "analysis_year": 2020 + (i % 3),
            "has_nih_funding": False,
            "has_industry_funding": False,
            "has_formal_capture": True,
        }
        for i in range(10, 20)
    ]

    df = pd.DataFrame(mock_data)
    engine = TrialClusteringEngine(n_clusters=2, use_embeddings=False)
    clustered_df, summaries = engine.fit_transform(df)

    assert "cluster_id" in clustered_df.columns
    assert "cluster_label" in clustered_df.columns
    assert len(summaries) == 2
    assert all("top_keywords" in info for info in summaries.values())


def test_concordance_evaluation():
    mock_study = {
        "nct_id": "NCT99999999",
        "eligibility_sex": "ALL",
        "gender_based": False,
        "gender_description": "",
        "eligibility_criteria_text": "Inclusion: Transgender women aged 18-45 on feminizing hormone therapy.",
        "brief_summary": "Study of estradiol levels in transgender women.",
    }
    res = evaluate_study_concordance(mock_study)
    assert not res["is_concordant"]
    assert "ALL_SEX_SINGLE_TRANS_GROUP_TEXT" in res["discordance_flags"]

"""
Rule-Based Benchmark Taxonomy for Transgender & Gender-Diverse Clinical Trials
Used as a reference baseline to evaluate and validate unsupervised clustering.
"""

import re
from typing import Any, Dict, List
import pandas as pd


class BenchmarkTaxonomyClassifier:
    """
    Categorizes trials into four primary biomedical & sociocultural domains:
    1. HIV / STI / MSM Syndemic Research
    2. Gender-Affirming Interventions (Hormonal, Surgical, Voice, Fertility)
    3. Routine & General Clinical Trials (explicit inclusion/exclusion)
    4. Mental Health, Substance Use, & Psychosocial Wellbeing
    """

    DOMAINS = [
        "HIV_STI_Syndemic",
        "Gender_Affirming_Care",
        "Mental_Health_Substance_Use",
        "General_Biomedical_Trial",
        "Other_Social_Observational",
    ]

    def classify_study(self, row: Dict[str, Any]) -> Dict[str, Any]:
        text = " ".join([
            str(row.get("brief_title", "") or ""),
            str(row.get("brief_summary", "") or ""),
            str(row.get("detailed_description", "") or ""),
            str(row.get("conditions", "") or ""),
            str(row.get("interventions", "") or ""),
            str(row.get("condition_mesh_terms", "") or ""),
            str(row.get("keywords", "") or ""),
        ]).lower()

        scores = {
            "HIV_STI_Syndemic": 0,
            "Gender_Affirming_Care": 0,
            "Mental_Health_Substance_Use": 0,
            "General_Biomedical_Trial": 0,
            "Other_Social_Observational": 0,
        }

        # 1. HIV / STI Syndemic Cues
        hiv_patterns = [
            r"\bhiv\b", r"\baids\b", r"\bprep\b", r"\bpep\b", r"\bantiretroviral\b",
            r"\bmsm\b", r"\bmen\s+who\s+have\s+sex\s+with\s+men\b", r"\bcondom\b",
            r"\bsti\b", r"\bsexually\s+transmitted\b", r"\bsyphilis\b", r"\bchlamydia\b", r"\bgonorrhea\b",
            r"\bviral\s+load\b", r"\bcd4\b"
        ]
        for p in hiv_patterns:
            if re.search(p, text):
                scores["HIV_STI_Syndemic"] += 2

        # 2. Gender Affirming Interventions
        affirming_patterns = [
            r"\bgender[\s-]?affirm\w*\b", r"\bhormone\s+therapy\b", r"\bcross[\s-]?sex\b",
            r"\btestosterone\b", r"\bestrogen\b", r"\bestradiol\b", r"\bpubert\w+\s+block\w*\b",
            r"\bgnrh\b", r"\bvaginoplast\w*\b", r"\bphalloplast\w*\b", r"\bmetoidioplast\w*\b",
            r"\bmastectomy\b", r"\bchest\s+masculiniz\w*\b", r"\btop\s+surgery\b", r"\bbottom\s+surgery\b",
            r"\bvoice\s+therap\w*\b", r"\bfacial\s+feminiz\w*\b", r"\bfertility\s+preservation\b"
        ]
        for p in affirming_patterns:
            if re.search(p, text):
                scores["Gender_Affirming_Care"] += 2

        # 3. Mental Health & Substance Use
        mh_patterns = [
            r"\bdepress\w*\b", r"\banxiet\w*\b", r"\bsuicid\w*\b", r"\bmental\s+health\b",
            r"\bminority\s+stress\b", r"\bpsychotherapy\b", r"\bcounseling\b", r"\bcbt\b",
            r"\bsubstance\s+use\b", r"\balcohol\b", r"\bopioid\b", r"\bcannabis\b",
            r"\baddiction\b", r"\bstigma\b", r"\bresilience\b"
        ]
        for p in mh_patterns:
            if re.search(p, text):
                scores["Mental_Health_Substance_Use"] += 1.5

        # 4. General Biomedical Trial Indicators (Oncology, Cardiology, Vaccine, Surgery)
        general_patterns = [
            r"\bcancer\b", r"\boncolog\w*\b", r"\bcarcinoma\b", r"\btumor\b",
            r"\bcardiovascular\b", r"\bdiabetes\b", r"\bhypertension\b", r"\bvaccine\b",
            r"\bphase\s+[1234]\b", r"\bpharmacokinetic\w*\b", r"\bdose\s+escalation\b"
        ]
        for p in general_patterns:
            if re.search(p, text):
                scores["General_Biomedical_Trial"] += 1

        # Determine primary domain
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        if max_score == 0:
            primary_domain = "Other_Social_Observational"
        else:
            primary_domain = max_domain

        return {
            "nct_id": row.get("nct_id", ""),
            "benchmark_primary_domain": primary_domain,
            "benchmark_domain_scores": scores,
        }

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records = [self.classify_study(row) for _, row in df.iterrows()]
        tax_df = pd.DataFrame(records)
        return tax_df

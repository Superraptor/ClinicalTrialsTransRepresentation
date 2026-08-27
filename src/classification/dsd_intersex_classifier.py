"""
DSD / Intersex Disentanglement and Cohort Specificity Classifier
Differentiates between:
1. TGD_CORE_SPECIFIC: Explicit focus on transgender, gender-diverse, gender dysphoria, or gender affirmation.
2. SGM_BROAD_INCLUSIVE: Broad Sexual and Gender Minority (LGBTQIA+) umbrella trials with trans inclusion.
3. DSD_INTERSEX_CONGENITAL: Disorders/Differences of Sex Development, CAH, Turner, Klinefelter without TGD focus.
4. ENDOCRINE_REPRODUCTIVE_OTHER: General endocrine/obstetric studies matching broad hormonal criteria.
"""

import logging
import re
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class CohortSpecificityClassifier:
    """Disentangles TGD core, SGM broad, DSD/intersex, and endocrine cohorts."""

    DSD_REGEX = re.compile(
        r"\b(?:"
        r"disorders?\s+of\s+sex\s+development|"
        r"differences?\s+of\s+sex\s+development|"
        r"intersex|"
        r"ambiguous\s+genitalia|"
        r"congenital\s+adrenal\s+hyperplasia|"
        r"androgen\s+insensitivity|"
        r"turner\s+syndrome|"
        r"klinefelter|"
        r"5-alpha\s+reductase\s+deficiency|"
        r"feminizing\s+genitoplasty\s+in\s+children|"
        r"gonadal\s+dysgenesis"
        r")\b",
        re.IGNORECASE,
    )

    TGD_CORE_REGEX = re.compile(
        r"\b(?:"
        r"transgender|"
        r"trans\s+woman|trans\s+women|trans\s+man|trans\s+men|"
        r"transsexual|transsexualism|"
        r"gender\s+dysphoria|gender\s+incongruence|gender\s+identity\s+disorder|"
        r"gender\s+affirm\w*|gender-affirming|"
        r"gender\s+diverse|gender\s+non-conforming|"
        r"non-binary|nonbinary|genderqueer|"
        r"chest\s+masculinization|vaginoplasty|phalloplasty|metoidioplasty|"
        r"puberty\s+suppression|pubertal\s+block\w*"
        r")\b",
        re.IGNORECASE,
    )

    SGM_BROAD_REGEX = re.compile(
        r"\b(?:"
        r"sexual\s+and\s+gender\s+minorit\w*|"
        r"sgm|lgbt\w*|2slgbt\w*|"
        r"men\s+who\s+have\s+sex\s+with\s+men|msm|"
        r"queer"
        r")\b",
        re.IGNORECASE,
    )

    def classify_study(self, study_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies a study record into primary population focus and flags."""
        text_corpus = " ".join([
            str(study_dict.get("brief_title", "") or ""),
            str(study_dict.get("official_title", "") or ""),
            str(study_dict.get("brief_summary", "") or ""),
            str(study_dict.get("detailed_description", "") or ""),
            str(study_dict.get("condition_mesh_terms", "") or ""),
            str(study_dict.get("intervention_mesh_terms", "") or ""),
            str(study_dict.get("eligibility_criteria_text", "") or ""),
        ])

        has_tgd = bool(self.TGD_CORE_REGEX.search(text_corpus))
        has_dsd = bool(self.DSD_REGEX.search(text_corpus))
        has_sgm = bool(self.SGM_BROAD_REGEX.search(text_corpus))

        # Determine primary classification
        if has_tgd:
            if has_dsd:
                cohort_type = "TGD_DSD_OVERLAP"
            elif has_sgm:
                cohort_type = "TGD_SGM_COMBINED"
            else:
                cohort_type = "TGD_CORE_SPECIFIC"
        elif has_dsd:
            cohort_type = "DSD_INTERSEX_CONGENITAL"
        elif has_sgm:
            cohort_type = "SGM_BROAD_ONLY"
        else:
            cohort_type = "ENDOCRINE_REPRODUCTIVE_OTHER"

        return {
            "nct_id": study_dict.get("nct_id", ""),
            "cohort_focus_type": cohort_type,
            "is_tgd_core": has_tgd,
            "is_dsd_intersex": has_dsd,
            "is_sgm_broad": has_sgm,
        }

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies an entire cohort dataframe."""
        results = [self.classify_study(row.to_dict()) for _, row in df.iterrows()]
        return pd.DataFrame(results)

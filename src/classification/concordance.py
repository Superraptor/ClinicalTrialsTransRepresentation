"""
Concordance and Structural Ambiguity Analysis
Evaluates alignment and discrepancies between formal structured eligibility fields
(sex, genderBased, genderDescription) and unstructured free-text eligibility rules.
"""

import logging
import re
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_study_concordance(row: Dict[str, Any], eligibility_rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Evaluates discordance between formal fields and textual criteria for a single study.
    """
    nct_id = row.get("nct_id", "")
    elig_sex = str(row.get("eligibility_sex", "ALL") or "ALL").upper()
    gender_based = bool(row.get("gender_based", False))
    gender_desc = str(row.get("gender_description", "") or "").lower()
    
    text_content = " ".join([
        str(row.get("eligibility_criteria_text", "") or ""),
        str(row.get("brief_summary", "") or ""),
        str(row.get("brief_title", "") or ""),
    ]).lower()

    # Specific demographic cues in text
    mentions_trans_women = bool(re.search(r"\b(?:trans(?:gender)?\s*wom[ae]n|transfeminine|mtf|trans(?:gender)?\s*female)\b", text_content))
    mentions_trans_men = bool(re.search(r"\b(?:trans(?:gender)?\s*m[ae]n|transmasculine|ftm|trans(?:gender)?\s*male)\b", text_content))
    mentions_nonbinary = bool(re.search(r"\b(?:non[\s-]?binary|gender[\s-]?queer|gender[\s-]?diverse|gender[\s-]?fluid)\b", text_content))
    mentions_natal_female = bool(re.search(r"\b(?:natal\s+females?|assigned\s+female\s+at\s+birth|afab|birth[\s-]assigned\s+female)\b", text_content))
    mentions_natal_male = bool(re.search(r"\b(?:natal\s+males?|assigned\s+male\s+at\s+birth|amab|birth[\s-]assigned\s+male)\b", text_content))
    mentions_msm_trans = bool(re.search(r"\b(?:msm|men\s+who\s+have\s+sex\s+with\s+men)\b", text_content) and mentions_trans_women)

    discordance_flags: List[str] = []

    # Case 1: Sex marked as ALL, but text specifically targets single trans gender group
    if elig_sex == "ALL":
        if (mentions_trans_women and not mentions_trans_men) or (mentions_trans_men and not mentions_trans_women):
            discordance_flags.append("ALL_SEX_SINGLE_TRANS_GROUP_TEXT")

    # Case 2: Binary Sex Marked (FEMALE or MALE) but genderBased is FALSE when trans populations are studied
    if elig_sex in ("FEMALE", "MALE") and not gender_based and (mentions_trans_women or mentions_trans_men):
        discordance_flags.append("BINARY_SEX_RESTRICTION_GENDERBASED_FALSE")

    # Case 3: Sex=FEMALE but study targets Trans Women (who are biologically AMAB) or Sex=MALE but study targets Trans Men (AFAB)
    if elig_sex == "FEMALE" and (mentions_trans_women or mentions_natal_male):
        discordance_flags.append("SEX_FEMALE_WITH_AMAB_OR_TRANS_WOMEN")
    if elig_sex == "MALE" and (mentions_trans_men or mentions_natal_female):
        discordance_flags.append("SEX_MALE_WITH_AFAB_OR_TRANS_MEN")

    # Case 4: Trans Women conflated with MSM in free text while sex=MALE or ALL
    if mentions_msm_trans:
        discordance_flags.append("MSM_TRANS_WOMEN_CO_INDEXED")

    # Case 5: genderBased=TRUE but empty gender_description
    if gender_based and not gender_desc.strip():
        discordance_flags.append("GENDER_BASED_TRUE_NO_DESCRIPTION")

    # Case 6: Completely concordant or no discrepancy detected
    is_concordant = len(discordance_flags) == 0

    return {
        "nct_id": nct_id,
        "eligibility_sex": elig_sex,
        "gender_based": gender_based,
        "is_concordant": is_concordant,
        "discordance_flags": ", ".join(discordance_flags) if discordance_flags else "CONCORDANT",
        "discordance_count": len(discordance_flags),
        "mentions_trans_women": mentions_trans_women,
        "mentions_trans_men": mentions_trans_men,
        "mentions_nonbinary": mentions_nonbinary,
        "mentions_natal_female": mentions_natal_female,
        "mentions_natal_male": mentions_natal_male,
        "mentions_msm_trans": mentions_msm_trans,
    }


def compute_concordance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes discordance indicators for an entire cohort dataframe.
    """
    results = [evaluate_study_concordance(row) for _, row in df.iterrows()]
    return pd.DataFrame(results)

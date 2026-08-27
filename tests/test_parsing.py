"""
Unit tests for parsing and extraction logic.
"""

import pytest
from src.parsing.trial_parser import parse_study_record, extract_year
from src.parsing.eligibility_parser import split_inclusion_exclusion, split_into_rules, parse_eligibility_rules
from src.parsing.term_extractor import TermExtractor


def test_extract_year():
    assert extract_year("2021-05-14") == 2021
    assert extract_year("December 2018") == 2018
    assert extract_year("2015-01") == 2015
    assert extract_year(None) is None


def test_split_inclusion_exclusion():
    text = """
    Inclusion Criteria:
    * Age >= 18
    * Transgender woman on hormone therapy
    
    Exclusion Criteria:
    * Active pregnancy
    * Severe cardiac disease
    """
    incl, excl = split_inclusion_exclusion(text)
    assert "Transgender woman" in incl
    assert "Active pregnancy" in excl


def test_split_into_rules():
    raw_section = """
    1. Assigned female at birth (AFAB)
    2. Age between 18 and 65
    * Currently on gender-affirming testosterone
    - No prior oophorectomy
    """
    rules = split_into_rules(raw_section)
    assert len(rules) == 4
    assert "Assigned female at birth (AFAB)" in rules[0]
    assert "gender-affirming testosterone" in rules[2]


def test_term_extractor_affirmative_and_historical():
    extractor = TermExtractor()
    sample_text = "This study evaluates vaginoplasty and facial feminization in transgender women with gender dysphoria."
    matches = extractor.extract_matches_from_text(sample_text, "NCT00000001", "brief_summary")
    
    matched_terms = {m["matched_text"].lower() for m in matches}
    categories = {m["category"] for m in matches}

    assert any("vaginoplast" in t for t in matched_terms)
    assert any("transgender women" in t or "transgender" in t for t in matched_terms)
    assert "diagnostic_contemporary" in categories or "identity_affirmative" in categories


def test_acronym_guards():
    extractor = TermExtractor()
    
    # Valid FTM context
    valid_text = "Inclusion criteria: FTM transgender men receiving testosterone therapy."
    matches_valid = extractor.extract_matches_from_text(valid_text, "NCT00000002", "eligibility_criteria_text")
    assert any(m["matched_text"] == "FTM" for m in matches_valid)

    # False positive FTM context (e.g. fractional shortening / first time mother)
    invalid_text = "Study in first time mothers (FTM) assessing fetal ultrasound."
    matches_invalid = extractor.extract_matches_from_text(invalid_text, "NCT00000003", "brief_summary")
    assert not any(m["matched_text"] == "FTM" for m in matches_invalid)

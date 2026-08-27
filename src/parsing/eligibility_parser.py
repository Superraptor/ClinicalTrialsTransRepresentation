"""
Eligibility Criteria Parser
Splits raw eligibility text into discrete Inclusion and Exclusion rule items
and tags sex/gender/trans-related constraints.
"""

import re
from typing import Any, Dict, List, Tuple


def split_inclusion_exclusion(criteria_text: str) -> Tuple[str, str]:
    """
    Separates raw criteria text into inclusion and exclusion text blocks.
    """
    if not criteria_text:
        return "", ""

    text = criteria_text.strip()
    
    # Common headers
    incl_match = re.search(r"(?i)\binclusion\s+criteria\b:?", text)
    excl_match = re.search(r"(?i)\bexclusion\s+criteria\b:?", text)

    if incl_match and excl_match:
        if incl_match.start() < excl_match.start():
            incl_text = text[incl_match.end():excl_match.start()].strip()
            excl_text = text[excl_match.end():].strip()
        else:
            excl_text = text[excl_match.end():incl_match.start()].strip()
            incl_text = text[incl_match.end():].strip()
        return incl_text, excl_text
    elif incl_match:
        return text[incl_match.end():].strip(), ""
    elif excl_match:
        return "", text[excl_match.end():].strip()
    else:
        # Default to treating all as inclusion if no header
        return text, ""


def split_into_rules(section_text: str) -> List[str]:
    """
    Splits a criteria text section into individual bulleted or numbered rules.
    """
    if not section_text:
        return []

    lines = section_text.splitlines()
    rules: List[str] = []
    current_rule: List[str] = []

    # Pattern for bullet / numbered list item starters
    bullet_pattern = re.compile(r"^\s*(?:[-*•–—+]|\d+[\.\)]|[a-zA-Z][\.\)])\s+")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_rule:
                rules.append(" ".join(current_rule))
                current_rule = []
            continue

        if bullet_pattern.match(line):
            if current_rule:
                rules.append(" ".join(current_rule))
            # Remove bullet marker
            clean_line = bullet_pattern.sub("", line).strip()
            current_rule = [clean_line]
        else:
            if current_rule:
                current_rule.append(stripped)
            else:
                current_rule = [stripped]

    if current_rule:
        rules.append(" ".join(current_rule))

    # Filter out empty or trivially short lines
    return [r.strip() for r in rules if len(r.strip()) > 3]


def parse_eligibility_rules(nct_id: str, criteria_text: str) -> List[Dict[str, Any]]:
    """
    Parses full eligibility criteria into structured rule dictionaries.
    """
    incl_text, excl_text = split_inclusion_exclusion(criteria_text)
    
    parsed_rules: List[Dict[str, Any]] = []

    for rule_idx, rule in enumerate(split_into_rules(incl_text), 1):
        parsed_rules.append({
            "nct_id": nct_id,
            "rule_type": "INCLUSION",
            "rule_index": rule_idx,
            "rule_text": rule,
        })

    for rule_idx, rule in enumerate(split_into_rules(excl_text), 1):
        parsed_rules.append({
            "nct_id": nct_id,
            "rule_type": "EXCLUSION",
            "rule_index": rule_idx,
            "rule_text": rule,
        })

    return parsed_rules

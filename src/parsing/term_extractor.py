"""
Term and Concept Extractor
Extracts, classifies, and annotates occurrences of TGD terms across formal and informal trial fields.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

logger = logging.getLogger(__name__)


class TermExtractor:
    """Extracts and annotates TGD concepts from structured and unstructured trial fields."""

    FORMAL_FIELDS = {
        "condition_mesh_terms",
        "condition_mesh_ids",
        "intervention_mesh_terms",
        "intervention_mesh_ids",
        "browse_leaf_terms",
        "keywords",
        "gender_based",
        "gender_description",
    }

    INFORMAL_FIELDS = {
        "brief_title",
        "official_title",
        "brief_summary",
        "detailed_description",
        "eligibility_criteria_text",
        "inclusion_criteria",
        "exclusion_criteria",
        "outcomes_text",
        "conditions",
        "interventions",
    }

    def __init__(
        self,
        lexicon_path: str = "config/lexicon.yaml",
        mesh_path: str = "config/mesh_terms.yaml",
    ):
        self.lexicon_path = Path(lexicon_path)
        self.mesh_path = Path(mesh_path)
        
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self.acronym_guards: Dict[str, Dict[str, List[str]]] = {}
        self.exclusion_patterns: List[re.Pattern] = []
        self.mesh_descriptors: List[Dict[str, Any]] = []

        self._load_configs()

    def _load_configs(self):
        if self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.categories = data.get("categories", {})
                self.acronym_guards = data.get("acronym_guards", {})
                
                raw_exclusions = data.get("false_positive_exclusions", [])
                self.exclusion_patterns = [re.compile(p, re.IGNORECASE) for p in raw_exclusions]

                for cat_name, cat_data in self.categories.items():
                    self.compiled_patterns[cat_name] = []
                    for pat in cat_data.get("patterns", []):
                        try:
                            compiled = re.compile(pat, re.IGNORECASE)
                            self.compiled_patterns[cat_name].append((compiled, pat))
                        except re.error as e:
                            logger.error(f"Invalid regex in {cat_name}: {pat} ({e})")

        if self.mesh_path.exists():
            with open(self.mesh_path, "r", encoding="utf-8") as f:
                mesh_data = yaml.safe_load(f)
                self.mesh_descriptors = mesh_data.get("mesh_descriptors", [])

    def _validate_acronym_context(self, acronym: str, text: str, match_span: Tuple[int, int]) -> bool:
        """Verifies if an acronym match has sufficient supportive context."""
        guard = self.acronym_guards.get(acronym.upper())
        if not guard:
            return True

        start = max(0, match_span[0] - 150)
        end = min(len(text), match_span[1] + 150)
        local_window = text[start:end].lower()

        # Check negative/exclusion context
        for excl in guard.get("exclude_contexts", []):
            if excl.lower() in local_window:
                return False

        # Check required positive context
        reqs = guard.get("require_contexts", [])
        if reqs and not any(r.lower() in local_window for r in reqs):
            return False

        return True

    def extract_matches_from_text(
        self,
        text: str,
        nct_id: str,
        field_name: str,
    ) -> List[Dict[str, Any]]:
        """Scans a single text string against all category patterns."""
        if not text or not isinstance(text, str):
            return []

        field_type = "FORMAL" if field_name in self.FORMAL_FIELDS else "INFORMAL"
        matches: List[Dict[str, Any]] = []

        for cat_name, pattern_list in self.compiled_patterns.items():
            for regex, raw_pat in pattern_list:
                for match in regex.finditer(text):
                    matched_str = match.group(0)
                    span = match.span()

                    # Guard against acronym false positives
                    if matched_str.upper() in self.acronym_guards:
                        if not self._validate_acronym_context(matched_str, text, span):
                            continue

                    # Context snippet
                    snippet_start = max(0, span[0] - 40)
                    snippet_end = min(len(text), span[1] + 40)
                    context_snippet = text[snippet_start:snippet_end].replace("\n", " ").strip()

                    matches.append({
                        "nct_id": nct_id,
                        "field_name": field_name,
                        "field_type": field_type,
                        "category": cat_name,
                        "pattern": raw_pat,
                        "matched_text": matched_str,
                        "char_start": span[0],
                        "char_end": span[1],
                        "context_snippet": context_snippet,
                    })

        return matches

    def evaluate_study_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a flattened study record for formal and informal TGD capture,
        returning match records, formal/informal flags, and matched category set.
        """
        nct_id = record.get("nct_id", "")
        all_matches: List[Dict[str, Any]] = []

        # 1. Evaluate informal free text fields
        text_fields = [
            ("brief_title", record.get("brief_title", "")),
            ("official_title", record.get("official_title", "")),
            ("brief_summary", record.get("brief_summary", "")),
            ("detailed_description", record.get("detailed_description", "")),
            ("eligibility_criteria_text", record.get("eligibility_criteria_text", "")),
            ("outcomes_text", record.get("outcomes_text", "")),
            ("conditions", record.get("conditions", "")),
            ("interventions", record.get("interventions", "")),
        ]

        for field_name, val in text_fields:
            if val:
                all_matches.extend(self.extract_matches_from_text(val, nct_id, field_name))

        # 2. Evaluate formal fields
        formal_text_fields = [
            ("keywords", record.get("keywords", "")),
            ("gender_description", record.get("gender_description", "")),
            ("condition_mesh_terms", record.get("condition_mesh_terms", "")),
            ("intervention_mesh_terms", record.get("intervention_mesh_terms", "")),
            ("browse_leaf_terms", record.get("browse_leaf_terms", "")),
        ]
        for field_name, val in formal_text_fields:
            if val:
                all_matches.extend(self.extract_matches_from_text(val, nct_id, field_name))

        # 3. Check MeSH descriptor explicit ID matches
        cond_mesh_ids = set((record.get("condition_mesh_ids", "") or "").split(", "))
        intr_mesh_ids = set((record.get("intervention_mesh_ids", "") or "").split(", "))
        all_mesh_ids = cond_mesh_ids | intr_mesh_ids

        matched_mesh_names = []
        for desc in self.mesh_descriptors:
            d_id = desc["descriptor_id"]
            if d_id in all_mesh_ids:
                matched_mesh_names.append(desc["descriptor_name"])
                all_matches.append({
                    "nct_id": nct_id,
                    "field_name": "mesh_descriptor_id",
                    "field_type": "FORMAL",
                    "category": "mesh_controlled_vocabulary",
                    "pattern": d_id,
                    "matched_text": desc["descriptor_name"],
                    "char_start": 0,
                    "char_end": len(desc["descriptor_name"]),
                    "context_snippet": f"MeSH ID: {d_id} ({desc['descriptor_name']})",
                })

        # Formal capture indicators
        has_formal_mesh = len(matched_mesh_names) > 0
        has_formal_gender_based = bool(record.get("gender_based", False))
        has_formal_keywords = any(m["field_name"] == "keywords" for m in all_matches)
        
        has_formal_capture = has_formal_mesh or has_formal_gender_based or has_formal_keywords
        has_informal_capture = any(m["field_type"] == "INFORMAL" for m in all_matches)

        matched_categories = list({m["category"] for m in all_matches})

        return {
            "nct_id": nct_id,
            "has_formal_capture": has_formal_capture,
            "has_informal_capture": has_informal_capture,
            "formal_only": has_formal_capture and not has_informal_capture,
            "informal_only": has_informal_capture and not has_formal_capture,
            "dual_formal_informal": has_formal_capture and has_informal_capture,
            "has_formal_mesh": has_formal_mesh,
            "matched_mesh_terms": ", ".join(matched_mesh_names),
            "matched_categories": ", ".join(sorted(matched_categories)),
            "total_match_count": len(all_matches),
            "matches": all_matches,
        }

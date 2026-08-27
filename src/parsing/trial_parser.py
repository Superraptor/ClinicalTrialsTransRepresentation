"""
Trial Parser for ClinicalTrials.gov API v2 JSON
Extracts and flattens hierarchical study JSON into structured analytical records.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_year(date_str: Optional[str]) -> Optional[int]:
    """Extract a 4-digit year from date string (YYYY-MM-DD, YYYY-MM, or Month YYYY)."""
    if not date_str:
        return None
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(date_str))
    return int(match.group(1)) if match else None


def parse_study_record(study_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a raw API v2 study JSON into a flattened analytical dictionary.
    """
    proto = study_raw.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    design_mod = proto.get("designModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    cond_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    desc_mod = proto.get("descriptionModule", {})
    
    # Derived section for MeSH
    derived = study_raw.get("derivedSection", {})
    cond_browse = derived.get("conditionBrowseModule", {})
    intr_browse = derived.get("interventionBrowseModule", {})

    # 1. Identification
    nct_id = id_mod.get("nctId", "")
    brief_title = id_mod.get("briefTitle", "") or ""
    official_title = id_mod.get("officialTitle", "") or ""
    acronym = id_mod.get("acronym", "") or ""
    org_study_id = id_mod.get("orgStudyIdInfo", {}).get("id", "") or ""

    # 2. Status & Dates
    overall_status = status_mod.get("overallStatus", "") or ""
    
    start_date_struct = status_mod.get("startDateStruct", {})
    start_date = start_date_struct.get("date", "")
    start_year = extract_year(start_date)

    submit_date = status_mod.get("studyFirstSubmitDate", "")
    submit_year = extract_year(submit_date)

    post_date_struct = status_mod.get("studyFirstPostDateStruct", {})
    post_date = post_date_struct.get("date", "")
    post_year = extract_year(post_date)

    comp_date_struct = status_mod.get("completionDateStruct", {})
    completion_date = comp_date_struct.get("date", "")
    completion_year = extract_year(completion_date)

    # Primary analysis year priority: post_year -> submit_year -> start_year
    analysis_year = post_year or submit_year or start_year

    # 3. Sponsorship & Funding
    lead_sponsor = sponsor_mod.get("leadSponsor", {})
    lead_sponsor_name = lead_sponsor.get("name", "") or ""
    lead_sponsor_class = lead_sponsor.get("class", "") or ""  # e.g., NIH, FED, INDUSTRY, OTHER

    collaborators = [c.get("name", "") for c in sponsor_mod.get("collaborators", []) if c.get("name")]
    
    # Funder classification
    funder_type = lead_sponsor_class
    has_nih_funding = (lead_sponsor_class == "NIH") or any("NIH" in c or "National Institutes of Health" in c for c in collaborators)
    has_industry_funding = (lead_sponsor_class == "INDUSTRY") or any("INC" in c.upper() or "PHARMA" in c.upper() for c in collaborators)

    # 4. Design & Phase
    study_type = design_mod.get("studyType", "") or ""  # INTERVENTIONAL, OBSERVATIONAL, EXPANDED_ACCESS
    phases = design_mod.get("phases", []) or []
    phase_str = ", ".join(phases) if phases else "NOT_APPLICABLE"
    
    design_info = design_mod.get("designInfo", {})
    allocation = design_info.get("allocation", "") or ""
    intervention_model = design_info.get("interventionModel", "") or ""
    primary_purpose = design_info.get("primaryPurpose", "") or ""
    masking_info = design_info.get("maskingInfo", {})
    masking = masking_info.get("masking", "") or ""

    # Enrollment
    enrollment_info = design_mod.get("enrollmentInfo", {})
    enrollment_count = enrollment_info.get("count")
    enrollment_type = enrollment_info.get("type", "")

    # 5. Formal Eligibility Fields
    elig_sex = str(elig_mod.get("sex", "ALL") or "ALL")  # ALL, FEMALE, MALE
    gender_based = bool(elig_mod.get("genderBased", False))
    gender_description = str(elig_mod.get("genderDescription", "") or "")
    min_age = str(elig_mod.get("minimumAge", "") or "")
    max_age = str(elig_mod.get("maximumAge", "") or "")
    healthy_volunteers = str(elig_mod.get("healthyVolunteers", "") or "")
    std_ages = elig_mod.get("stdAges", []) or []

    # 6. Conditions & Interventions
    conditions_list = cond_mod.get("conditions", []) or []
    keywords_list = cond_mod.get("keywords", []) or []

    interventions_raw = arms_mod.get("interventions", []) or []
    intervention_names = [i.get("name", "") for i in interventions_raw if i.get("name")]
    intervention_types = [i.get("type", "") for i in interventions_raw if i.get("type")]

    # 7. MeSH Headings & Browse Leaves
    cond_mesh_list = cond_browse.get("meshes", []) or []
    cond_mesh_terms = [m.get("term", "") for m in cond_mesh_list if m.get("term")]
    cond_mesh_ids = [m.get("id", "") for m in cond_mesh_list if m.get("id")]

    intr_mesh_list = intr_browse.get("meshes", []) or []
    intr_mesh_terms = [m.get("term", "") for m in intr_mesh_list if m.get("term")]
    intr_mesh_ids = [m.get("id", "") for m in intr_mesh_list if m.get("id")]

    leaf_list = cond_browse.get("browseLeaves", []) + intr_browse.get("browseLeaves", [])
    browse_leaf_terms = list({l.get("name", "") for l in leaf_list if l.get("name")})

    # 8. Free-Text Content
    brief_summary = desc_mod.get("briefSummary", "") or ""
    detailed_description = desc_mod.get("detailedDescription", "") or ""
    eligibility_criteria_text = elig_mod.get("eligibilityCriteria", "") or ""

    # Primary and Secondary Outcome Measures
    primary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", []) if o.get("measure")]
    secondary_outcomes = [o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", []) if o.get("measure")]
    outcomes_text = " | ".join(primary_outcomes + secondary_outcomes)

    # Has posted results
    has_results = bool(study_raw.get("hasResults", False))

    # Retrieval queries provenance
    retrieval_queries = study_raw.get("_retrieval_queries", [])

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "acronym": acronym,
        "org_study_id": org_study_id,
        "overall_status": overall_status,
        "start_date": start_date,
        "start_year": start_year,
        "submit_date": submit_date,
        "submit_year": submit_year,
        "post_date": post_date,
        "post_year": post_year,
        "completion_date": completion_date,
        "completion_year": completion_year,
        "analysis_year": analysis_year,
        "lead_sponsor_name": lead_sponsor_name,
        "lead_sponsor_class": lead_sponsor_class,
        "has_nih_funding": has_nih_funding,
        "has_industry_funding": has_industry_funding,
        "study_type": study_type,
        "phases": phase_str,
        "allocation": allocation,
        "intervention_model": intervention_model,
        "primary_purpose": primary_purpose,
        "masking": masking,
        "enrollment_count": enrollment_count,
        "enrollment_type": enrollment_type,
        "eligibility_sex": elig_sex,
        "gender_based": gender_based,
        "gender_description": gender_description,
        "minimum_age": min_age,
        "maximum_age": max_age,
        "healthy_volunteers": healthy_volunteers,
        "std_ages": ", ".join(std_ages),
        "conditions": ", ".join(conditions_list),
        "interventions": ", ".join(intervention_names),
        "intervention_types": ", ".join(set(intervention_types)),
        "keywords": ", ".join(keywords_list),
        "condition_mesh_terms": ", ".join(cond_mesh_terms),
        "condition_mesh_ids": ", ".join(cond_mesh_ids),
        "intervention_mesh_terms": ", ".join(intr_mesh_terms),
        "intervention_mesh_ids": ", ".join(intr_mesh_ids),
        "browse_leaf_terms": ", ".join(browse_leaf_terms),
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        "eligibility_criteria_text": eligibility_criteria_text,
        "outcomes_text": outcomes_text,
        "has_results": has_results,
        "retrieval_queries": ", ".join(retrieval_queries) if isinstance(retrieval_queries, list) else str(retrieval_queries),
    }

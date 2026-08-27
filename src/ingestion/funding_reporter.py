"""
NIH RePORTER Integration and Funding Amount Analytics
Extracts NIH grant identifiers and queries the NIH RePORTER API v2 in efficient batches
to retrieve exact dollar amounts ($ USD), administering NIH institutes, PIs, and recipient institutions.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class FundingReporterClient:
    """Client for retrieving grant dollar amounts and institutional funding from NIH RePORTER API v2."""

    REPORTER_API_URL = "https://api.reporter.nih.gov/v2/projects/search"

    # Regex for standard NIH Grant Numbers (e.g., R01MH112345, U01AI123456, K23HD098765, 5R01DA043210-02)
    NIH_GRANT_REGEX = re.compile(r"\b(?:\d?[A-Z]{1,3}\d{1,2})?([A-Z]{2}\d{5,8})(?:-\d{2}[A-Z\d]*)?\b", re.IGNORECASE)

    def __init__(self, cache_dir: str = "data/raw/funding"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClinicalTrialsTransRepresentationFundingAnalytics/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def extract_grant_numbers_from_study(self, study_dict: Dict[str, Any]) -> List[str]:
        """Extracts potential NIH grant numbers from IDs and text fields."""
        candidates = set()
        
        # Check org_study_id and secondary IDs
        for field in ["org_study_id", "acronym"]:
            val = str(study_dict.get(field, "") or "")
            for m in self.NIH_GRANT_REGEX.finditer(val):
                candidates.add(m.group(0).upper().strip())

        # Check brief summary and detailed description
        for field in ["brief_summary", "detailed_description"]:
            val = str(study_dict.get(field, "") or "")
            for m in self.NIH_GRANT_REGEX.finditer(val):
                matched_str = m.group(0).upper().strip()
                if any(prefix in matched_str for prefix in ["R01", "R21", "U01", "K23", "K01", "K24", "R34", "P30", "F31", "F32", "UG3", "UH3", "DP2"]):
                    candidates.add(matched_str)

        return sorted(list(candidates))

    def query_nih_reporter_batch(
        self,
        nct_ids: Optional[List[str]] = None,
        grant_nums: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Queries NIH RePORTER API for a batch of NCT IDs or associated grant numbers.
        """
        criteria: Dict[str, Any] = {}
        if nct_ids:
            criteria["clinical_trials"] = nct_ids
        if grant_nums:
            criteria["project_nums"] = grant_nums

        if not criteria:
            return []

        payload = {
            "criteria": criteria,
            "include_fields": [
                "ProjectNum", "ApplId", "FiscalYear", "AwardAmount", "DirectCostAmt", "IndirectCostAmt",
                "ProjectTitle", "ContactPiName", "OrgName", "OrgCity", "OrgState", "OrgCountry",
                "AgencyIcAdmin", "FullStudySection", "ClinicalTrials"
            ],
            "limit": limit,
            "offset": 0,
        }

        try:
            time.sleep(0.3)
            resp = self.session.post(self.REPORTER_API_URL, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
            else:
                logger.warning(f"NIH RePORTER API returned HTTP {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error querying NIH RePORTER batch: {e}")
            return []

    def fetch_cohort_funding(
        self,
        studies_df: pd.DataFrame,
        output_parquet: str = "data/processed/funding_awards.parquet",
        batch_size: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves NIH funding records for all candidate studies using batched queries.
        """
        cache_file = self.cache_dir / "nih_reporter_cache.json"
        cached_records: Dict[str, List[Dict[str, Any]]] = {}

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_records = json.load(f)
                    logger.info(f"Loaded {len(cached_records)} cached NIH RePORTER records.")
            except Exception as e:
                logger.warning(f"Failed to read funding cache: {e}")

        # Gather all NCT IDs and Grants
        all_ncts = studies_df["nct_id"].tolist()
        uncached_ncts = [n for n in all_ncts if n not in cached_records]

        # Gather explicit grants
        nct_to_grants: Dict[str, List[str]] = {}
        all_grant_candidates = set()
        for _, row in studies_df.iterrows():
            nct = row["nct_id"]
            grants = self.extract_grant_numbers_from_study(row.to_dict())
            if grants:
                nct_to_grants[nct] = grants
                all_grant_candidates.update(grants)

        if uncached_ncts:
            logger.info(f"Batch querying NIH RePORTER for {len(uncached_ncts)} NCT IDs in chunks of {batch_size}...")
            for i in range(0, len(uncached_ncts), batch_size):
                chunk = uncached_ncts[i : i + batch_size]
                results = self.query_nih_reporter_batch(nct_ids=chunk, limit=500)
                
                # Initialize empty for all in chunk
                for nct in chunk:
                    if nct not in cached_records:
                        cached_records[nct] = []

                # Distribute matched results to respective NCTs
                for res in results:
                    ct_list = res.get("clinical_trials", []) or res.get("ClinicalTrials", []) or []
                    for ct in ct_list:
                        c_id = ct.get("nct_id") or ct.get("NctId") or ct.get("id") or ""
                        if c_id in cached_records:
                            cached_records[c_id].append(res)

            # Also query any explicit grant numbers that didn't match by NCT
            grant_list = list(all_grant_candidates)
            if grant_list:
                logger.info(f"Querying NIH RePORTER for {len(grant_list)} candidate grant identifiers...")
                for i in range(0, len(grant_list), 50):
                    g_chunk = grant_list[i : i + 50]
                    g_results = self.query_nih_reporter_batch(grant_nums=g_chunk, limit=500)
                    for res in g_results:
                        pnum = str(res.get("project_num") or res.get("ProjectNum") or "")
                        # match back to NCT
                        for nct, g_candidates in nct_to_grants.items():
                            if any(g in pnum for g in g_candidates):
                                if nct not in cached_records:
                                    cached_records[nct] = []
                                if res not in cached_records[nct]:
                                    cached_records[nct].append(res)

        # Save updated cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_records, f, indent=2)

        # Format tabular outputs
        all_award_rows: List[Dict[str, Any]] = []
        study_funding_summary: List[Dict[str, Any]] = []

        for _, row in studies_df.iterrows():
            nct_id = row["nct_id"]
            results = cached_records.get(nct_id, [])

            total_study_funding = 0.0
            ic_list = set()
            org_list = set()
            fiscal_years = set()

            for res in results:
                award_amt = float(res.get("award_amount") or res.get("AwardAmount") or 0.0)
                fy = res.get("fiscal_year") or res.get("FiscalYear")
                ic = res.get("agency_ic_admin", {}).get("name") or str(res.get("AgencyIcAdmin", "") or "")
                org = res.get("org_name") or res.get("OrgName") or ""
                pi = res.get("contact_pi_name") or res.get("ContactPiName") or ""
                proj_num = res.get("project_num") or res.get("ProjectNum") or ""

                total_study_funding += award_amt
                if ic:
                    ic_list.add(ic)
                if org:
                    org_list.add(org)
                if fy:
                    fiscal_years.add(int(fy))

                all_award_rows.append({
                    "nct_id": nct_id,
                    "project_num": proj_num,
                    "fiscal_year": fy,
                    "award_amount_usd": award_amt,
                    "administering_ic": ic,
                    "recipient_org": org,
                    "contact_pi": pi,
                    "project_title": res.get("project_title") or res.get("ProjectTitle") or "",
                })

            study_funding_summary.append({
                "nct_id": nct_id,
                "has_linked_nih_grant": len(results) > 0,
                "nih_award_count": len(results),
                "total_nih_funding_usd": total_study_funding,
                "administering_nih_ics": ", ".join(sorted(list(ic_list))),
                "recipient_institutions": ", ".join(sorted(list(org_list))),
                "funding_fiscal_years": ", ".join([str(y) for y in sorted(list(fiscal_years))]),
            })

        awards_df = pd.DataFrame(all_award_rows)
        summary_df = pd.DataFrame(study_funding_summary)

        out_p = Path(output_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        if len(awards_df) > 0:
            for col in awards_df.select_dtypes(include=["object"]).columns:
                awards_df[col] = awards_df[col].fillna("").astype(str)
            awards_df.to_parquet(out_p, index=False)
            awards_df.to_csv(out_p.with_suffix(".csv"), index=False)

        summary_df.to_parquet(out_p.parent / "study_funding_summary.parquet", index=False)
        summary_df.to_csv(out_p.parent / "study_funding_summary.csv", index=False)

        logger.info(f"Funding analysis complete. Found {len(awards_df)} award records totaling ${awards_df['award_amount_usd'].sum():,.2f} USD." if len(awards_df) > 0 else "No linked awards.")
        return awards_df, summary_df

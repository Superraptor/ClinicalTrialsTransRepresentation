"""
Publication Extractor and PubMed / PMC / DOI Resolver
Extracts publication records from ClinicalTrials.gov referencesModule and resolves
PMIDs to PMCIDs, DOIs, publication titles, and journal metadata via NCBI E-utilities.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class PublicationExtractor:
    """Extracts and enriches publications linked to clinical trials."""

    EUTILS_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    EUTILS_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

    def __init__(self, cache_dir: str = "data/raw/publications", email: str = "trans-representation-study@research.local"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"ClinicalTrialsPubResolver/1.0 ({email})"})

    def extract_raw_references(self, raw_study: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts PMIDs, citations, and reference types from raw study JSON."""
        proto = raw_study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        nct_id = id_mod.get("nctId", "")
        
        refs_mod = proto.get("referencesModule", {})
        ref_list = refs_mod.get("references", []) or []

        extracted = []
        doi_regex = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")

        for idx, ref in enumerate(ref_list, 1):
            pmid = str(ref.get("pmid", "") or "").strip()
            ref_type = str(ref.get("type", "REFERENCE") or "REFERENCE")
            citation = str(ref.get("citation", "") or "")

            # Search for DOI in citation text
            doi_match = doi_regex.search(citation)
            doi = doi_match.group(0).rstrip(".,;") if doi_match else ""

            extracted.append({
                "nct_id": nct_id,
                "ref_index": idx,
                "pmid": pmid if pmid.isdigit() else "",
                "reference_type": ref_type,
                "citation": citation,
                "extracted_doi": doi,
            })

        # Also check seeAlsoLinks
        see_links = refs_mod.get("seeAlsoLinks", []) or []
        for s_idx, link in enumerate(see_links, 1):
            url = link.get("url", "")
            label = link.get("label", "")
            extracted.append({
                "nct_id": nct_id,
                "ref_index": len(ref_list) + s_idx,
                "pmid": "",
                "reference_type": "SEE_ALSO",
                "citation": f"{label}: {url}" if label else url,
                "extracted_doi": "",
            })

        return extracted

    def resolve_pmids_batch(self, pmids: List[str], chunk_size: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Queries NCBI E-utilities to resolve PMIDs to titles, journals, pub dates, PMCIDs, and DOIs.
        """
        valid_pmids = list({p for p in pmids if p and p.isdigit()})
        if not valid_pmids:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        cache_file = self.cache_dir / "ncbi_pmid_cache.json"

        # Load cache if available
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read NCBI cache: {e}")

        to_fetch = [p for p in valid_pmids if p not in results]
        logger.info(f"Resolving {len(to_fetch)} new PMIDs via NCBI E-utilities ({len(results)} cached)...")

        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i : i + chunk_size]
            id_str = ",".join(chunk)
            params = {
                "db": "pubmed",
                "id": id_str,
                "retmode": "json",
                "email": self.email,
            }

            try:
                time.sleep(0.35)
                resp = self.session.get(self.EUTILS_SUMMARY_URL, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    summary_result = data.get("result", {})
                    for pmid in chunk:
                        item = summary_result.get(pmid, {})
                        if not item or "error" in item:
                            results[pmid] = {
                                "pmid": pmid,
                                "title": "",
                                "journal": "",
                                "pub_date": "",
                                "pub_year": None,
                                "pmcid": "",
                                "doi": "",
                                "authors": "",
                            }
                            continue

                        # Extract article IDs (DOI, PMCID)
                        doi = ""
                        pmcid = ""
                        for art_id in item.get("articleids", []):
                            if art_id.get("idtype") == "doi":
                                doi = art_id.get("value", "")
                            elif art_id.get("idtype") == "pmc":
                                pmcid = art_id.get("value", "")

                        # Authors
                        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
                        
                        # Pub Year
                        pub_date = item.get("pubdate", "")
                        year_match = re.search(r"\b(19\d\d|20\d\d)\b", str(pub_date))
                        pub_year = int(year_match.group(1)) if year_match else None

                        results[pmid] = {
                            "pmid": pmid,
                            "title": item.get("title", ""),
                            "journal": item.get("source", ""),
                            "pub_date": pub_date,
                            "pub_year": pub_year,
                            "pmcid": pmcid,
                            "doi": doi,
                            "authors": ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                        }
                else:
                    logger.warning(f"NCBI E-utilities returned HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"Error resolving PMIDs chunk: {e}")

        # Save cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results

    def process_and_enrich_studies(
        self,
        raw_studies: List[Dict[str, Any]],
        cohort_nct_ids: Optional[Set[str]] = None,
        output_parquet: str = "data/processed/publications.parquet",
    ) -> pd.DataFrame:
        """
        Extracts references exclusively from the trial cohort, enriches with NCBI metadata,
        and saves the publication dataset.
        """
        if cohort_nct_ids is not None:
            raw_studies = [
                st for st in raw_studies
                if st.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "") in cohort_nct_ids
            ]

        all_refs = []
        all_pmids = set()

        for st in raw_studies:
            refs = self.extract_raw_references(st)
            for r in refs:
                if r["pmid"]:
                    all_pmids.add(r["pmid"])
            all_refs.extend(refs)

        # Resolve PMIDs via NCBI
        pmid_metadata = self.resolve_pmids_batch(list(all_pmids))

        # Merge resolved metadata
        enriched = []
        for r in all_refs:
            pmid = r["pmid"]
            meta = pmid_metadata.get(pmid, {})
            row = dict(r)
            row["resolved_title"] = meta.get("title", "")
            row["resolved_journal"] = meta.get("journal", "")
            row["resolved_pub_date"] = meta.get("pub_date", "")
            row["resolved_pub_year"] = meta.get("pub_year")
            row["pmcid"] = meta.get("pmcid", "")
            row["doi"] = meta.get("doi") or r["extracted_doi"]
            row["authors"] = meta.get("authors", "")
            row["is_open_access"] = bool(meta.get("pmcid"))
            enriched.append(row)

        pub_df = pd.DataFrame(enriched)
        
        out_p = Path(output_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        if len(pub_df) > 0:
            for col in pub_df.select_dtypes(include=["object"]).columns:
                pub_df[col] = pub_df[col].fillna("").astype(str)
            pub_df.to_parquet(out_p, index=False)
            pub_df.to_csv(out_p.with_suffix(".csv"), index=False)

        logger.info(f"Processed {len(pub_df)} total references ({len(all_pmids)} unique PMIDs). Saved to {out_p}")
        return pub_df

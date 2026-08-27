"""
API Client for ClinicalTrials.gov REST API v2
Handles pagination, rate limiting, retries, checkpointing, and raw JSON persistence.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ClinicalTrialsApiClient:
    """Client for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(
        self,
        base_url: str = BASE_URL,
        page_size: int = 100,
        rate_limit_delay_sec: float = 0.25,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
        request_timeout: int = 30,
        cache_dir: str = "data/raw/api_responses",
        user_agent: str = "ClinicalTrialsTransRepresentationResearch/1.0",
    ):
        self.base_url = base_url.rstrip("/")
        self.page_size = min(page_size, 1000)
        self.rate_limit_delay = rate_limit_delay_sec
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
        })

    def _request_with_retry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GET request with exponential backoff retry."""
        url = self.base_url
        attempt = 0
        backoff = self.rate_limit_delay

        while attempt < self.max_retries:
            attempt += 1
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, params=params, timeout=self.request_timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429). Retrying in {backoff:.2f}s... (Attempt {attempt}/{self.max_retries})")
                    time.sleep(backoff)
                    backoff *= self.backoff_factor
                elif response.status_code >= 500:
                    logger.warning(f"Server error ({response.status_code}). Retrying in {backoff:.2f}s... (Attempt {attempt}/{self.max_retries})")
                    time.sleep(backoff)
                    backoff *= self.backoff_factor
                else:
                    logger.error(f"API HTTP Error {response.status_code}: {response.text}")
                    response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                logger.warning(f"Request error: {e}. Retrying in {backoff:.2f}s... (Attempt {attempt}/{self.max_retries})")
                time.sleep(backoff)
                backoff *= self.backoff_factor

        raise RuntimeError(f"Failed to fetch data from {url} after {self.max_retries} attempts. Params: {params}")

    def search_studies(
        self,
        term: Optional[str] = None,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        mesh_term: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        max_studies: Optional[int] = None,
        save_pages: bool = True,
        query_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Paginates through search results for a given query parameter set.
        Returns a list of raw study objects (dict).
        """
        params: Dict[str, Any] = {
            "pageSize": self.page_size,
            "countTotal": "true",
            "format": "json",
        }
        
        if term:
            params["query.term"] = term
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if mesh_term:
            params["query.term"] = f"AREA[ConditionMeshTerm]{mesh_term} OR AREA[InterventionMeshTerm]{mesh_term}"
        if additional_params:
            params.update(additional_params)

        tag = query_tag or (term or condition or intervention or mesh_term or "all_studies")
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)[:50]
        
        all_studies: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        page_idx = 0
        total_count = None

        logger.info(f"Starting search for query tag '{safe_tag}' with params: {params}")

        while True:
            page_idx += 1
            current_params = dict(params)
            if page_token:
                current_params["pageToken"] = page_token

            data = self._request_with_retry(current_params)
            
            if total_count is None and "totalCount" in data:
                total_count = data["totalCount"]
                logger.info(f"Query '{safe_tag}' reported total matching studies: {total_count}")

            studies = data.get("studies", [])
            if not studies:
                break

            all_studies.extend(studies)
            
            if save_pages:
                page_file = self.cache_dir / f"{safe_tag}_page_{page_idx:04d}.json"
                with open(page_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Fetched page {page_idx} ({len(studies)} studies, accumulated: {len(all_studies)}/{total_count or '?'})")

            if max_studies and len(all_studies) >= max_studies:
                all_studies = all_studies[:max_studies]
                break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Completed search for '{safe_tag}'. Total retrieved: {len(all_studies)}")
        return all_studies

    def query_sweep(
        self,
        queries: List[str],
        mesh_terms: Optional[List[str]] = None,
        save_consolidated_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes a multi-query sweep across a list of terms and MeSH headings.
        Deduplicates results by nctId while tracking which query found each study.
        """
        dedup_studies: Dict[str, Dict[str, Any]] = {}
        query_provenance: Dict[str, Set[str]] = {}

        # 1. Sweep text queries
        for q in queries:
            tag = f"text_{q}"
            studies = self.search_studies(term=q, query_tag=tag)
            for st in studies:
                nct_id = st.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if not nct_id:
                    continue
                if nct_id not in dedup_studies:
                    dedup_studies[nct_id] = st
                    query_provenance[nct_id] = set()
                query_provenance[nct_id].add(f"text:{q}")

        # 2. Sweep MeSH terms
        if mesh_terms:
            for mesh in mesh_terms:
                tag = f"mesh_{mesh}"
                studies = self.search_studies(mesh_term=mesh, query_tag=tag)
                for st in studies:
                    nct_id = st.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                    if not nct_id:
                        continue
                    if nct_id not in dedup_studies:
                        dedup_studies[nct_id] = st
                        query_provenance[nct_id] = set()
                    query_provenance[nct_id].add(f"mesh:{mesh}")

        # Attach provenance to study dict
        for nct_id, st in dedup_studies.items():
            st["_retrieval_queries"] = sorted(list(query_provenance.get(nct_id, [])))

        logger.info(f"Query sweep completed. Unique studies collected: {len(dedup_studies)}")

        if save_consolidated_path:
            out_p = Path(save_consolidated_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(list(dedup_studies.values()), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved consolidated studies to {out_p}")

        return dedup_studies

    def get_study_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single study JSON by its NCT ID."""
        url = f"{self.base_url}/{nct_id}"
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, timeout=self.request_timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching study {nct_id}: {e}")
            return None

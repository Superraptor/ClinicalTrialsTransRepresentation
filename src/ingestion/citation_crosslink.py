"""
Citation Cross-Linkage Extractor (NCBI E-utilities elink & Crossref)
Discovers direct peer-to-peer citation linkages between scientific publications (PMIDs / DOIs)
that are part of the clinical trial representation graph.
Strictly filters out all external/indirect citations, retaining only internal graph edges.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class CitationCrossLinker:
    """Discovers and extracts direct citation relationships among cohort publications."""

    def __init__(
        self,
        cache_path: str = "data/raw/publications/citation_crosslinks_cache.json",
        user_agent: str = "ClinicalTrialsRepresentationResearch/1.0 (mailto:research@example.org)",
        batch_size: int = 100,
        rate_limit_delay_sec: float = 0.35,
    ):
        self.cache_path = Path(cache_path)
        self.user_agent = user_agent
        self.batch_size = batch_size
        self.rate_limit_delay_sec = rate_limit_delay_sec
        self.cache: Dict[str, List[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load citation cache: {e}")
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def fetch_citing_pmids_batch(self, pmids: List[str]) -> Dict[str, List[str]]:
        """
        Batch queries NCBI E-utilities elink for citing PMIDs (pubmed_pubmed_citedin).
        """
        uncached = [p for p in pmids if p not in self.cache]
        if not uncached:
            return {p: self.cache.get(p, []) for p in pmids}

        logger.info(f"Querying NCBI elink for {len(uncached)} uncached PMIDs in batches of {self.batch_size}...")

        for i in range(0, len(uncached), self.batch_size):
            chunk = uncached[i : i + self.batch_size]
            id_str = ",".join(chunk)
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id={id_str}&retmode=json"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                # Parse linksets
                for ls in data.get("linksets", []):
                    src_ids = ls.get("ids", [])
                    if not src_ids:
                        continue
                    src_pmid = str(src_ids[0])
                    citing_list = []
                    for db in ls.get("linksetdbs", []):
                        if db.get("linkname") == "pubmed_pubmed_citedin":
                            citing_list.extend([str(x) for x in db.get("links", [])])
                    self.cache[src_pmid] = citing_list

                # Ensure all in chunk are cached even if no citations found
                for p in chunk:
                    if p not in self.cache:
                        self.cache[p] = []

            except Exception as e:
                logger.warning(f"Error fetching elink batch starting at {i}: {e}")
                for p in chunk:
                    if p not in self.cache:
                        self.cache[p] = []

            time.sleep(self.rate_limit_delay_sec)

        self._save_cache()
        return {p: self.cache.get(p, []) for p in pmids}

    def discover_internal_cross_citations(
        self,
        pub_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extracts direct citations (Citing PMID -> Cited PMID) where BOTH papers belong to the cohort.
        """
        valid_pubs = pub_df[pub_df["pmid"] != ""].copy()
        cohort_pmids: Set[str] = set(valid_pubs["pmid"].astype(str))

        logger.info(f"Targeting internal cross-linkages among {len(cohort_pmids)} unique PMIDs in cohort...")
        self.fetch_citing_pmids_batch(list(cohort_pmids))

        # Build PMIDs mapping to metadata
        pmid_to_nct = valid_pubs.groupby("pmid")["nct_id"].apply(lambda s: list(set(s))).to_dict()
        pmid_to_journal = valid_pubs.set_index("pmid")["resolved_journal"].to_dict()
        pmid_to_year = valid_pubs.set_index("pmid")["resolved_pub_year"].to_dict()

        edges = []
        for cited_pmid, citing_list in self.cache.items():
            if cited_pmid not in cohort_pmids:
                continue
            for citing_pmid in citing_list:
                if citing_pmid in cohort_pmids and citing_pmid != cited_pmid:
                    edges.append({
                        "citing_pmid": citing_pmid,
                        "cited_pmid": cited_pmid,
                        "citing_ncts": pmid_to_nct.get(citing_pmid, []),
                        "cited_ncts": pmid_to_nct.get(cited_pmid, []),
                        "citing_journal": pmid_to_journal.get(citing_pmid, ""),
                        "cited_journal": pmid_to_journal.get(cited_pmid, ""),
                        "citing_year": pmid_to_year.get(citing_pmid, None),
                        "cited_year": pmid_to_year.get(cited_pmid, None),
                        "edge_type": "DIRECT_CITATION",
                    })

        edges_df = pd.DataFrame(edges)
        if len(edges_df) > 0:
            edges_df.drop_duplicates(subset=["citing_pmid", "cited_pmid"], inplace=True)

        logger.info(f"Discovered {len(edges_df)} direct internal cross-citation edges between PMIDs in the graph.")

        out_path = Path("data/processed/citation_crosslinks.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_parquet(out_path, index=False)
        edges_df.to_csv("data/processed/citation_crosslinks.csv", index=False)

        return edges_df

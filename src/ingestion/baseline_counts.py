"""
Baseline Denominator Collector
Fetches the total number of clinical trials registered on ClinicalTrials.gov per calendar year.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_annual_baseline_counts(
    start_year: int = 1999,
    end_year: int = 2026,
    output_path: str = "data/raw/baseline_counts.json",
    delay_sec: float = 0.3,
) -> Dict[int, int]:
    """
    Queries ClinicalTrials.gov API v2 to count total studies registered per year
    using StudyFirstPostDate ranges.
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    counts_by_year: Dict[int, int] = {}
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                counts_by_year = {int(k): int(v) for k, v in cached.items()}
                logger.info(f"Loaded {len(counts_by_year)} annual baseline counts from cache: {out_file}")
                return counts_by_year
        except Exception as e:
            logger.warning(f"Failed to read baseline cache: {e}. Re-fetching...")

    session = requests.Session()
    session.headers.update({"User-Agent": "ClinicalTrialsTransRepresentationResearch/1.0"})

    logger.info(f"Fetching annual baseline study counts for years {start_year}-{end_year}...")

    for year in range(start_year, end_year + 1):
        date_start = f"{year}-01-01"
        date_end = f"{year}-12-31"
        
        # ClinicalTrials.gov query syntax for date ranges
        query_filter = f"AREA[StudyFirstPostDate]RANGE[{date_start}, {date_end}]"
        params = {
            "query.term": query_filter,
            "pageSize": 1,
            "countTotal": "true",
            "format": "json",
        }

        try:
            time.sleep(delay_sec)
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("totalCount", 0)
                counts_by_year[year] = total
                logger.info(f"Year {year}: {total:,} total registered studies")
            else:
                logger.error(f"Failed to fetch baseline for {year}: HTTP {resp.status_code}")
                counts_by_year[year] = 0
        except Exception as e:
            logger.error(f"Error fetching baseline for {year}: {e}")
            counts_by_year[year] = 0

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(counts_by_year, f, indent=2)
    logger.info(f"Saved baseline counts to {out_file}")

    return counts_by_year

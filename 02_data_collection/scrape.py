# Updated 02_data_collection/scrape.py with metadata and layout info
import os
import json
import requests
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from sys import path

# Load config
path.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config, load_env

# ─── Logging setup ─────────────────────────────────────────────────────────────
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "scrape.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Generate a unique section ID

def make_section_id(url, idx):
    base = url.replace('https://','').replace('http://','').replace('/', '_')
    return f"{base}_p{idx}"


def scrape_websites():
    config = load_config()
    out_dir = Path(config["output"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    crawl_date = datetime.utcnow().isoformat() + 'Z'
    for url in config["data_sources"]["websites"]:
        logger.info(f"Fetching {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract paragraphs as sections
        sections = []
        for idx, p in enumerate(soup.find_all(["article","main","p"]), start=1):
            text = p.get_text(strip=True)
            if not text:
                continue
            section_id = make_section_id(url, idx)
            sections.append({
                "section_id": section_id,
                "text": text,
                "word_count": len(text.split()),
            })

        data = {
            "source": url,
            "crawl_date": crawl_date,
            "sections": sections
        }
        fname = url.replace("https://", "").replace("http://", "").replace("/", "_")
        out_path = out_dir / f"{fname}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved scraped data to {out_path}")


if __name__ == "__main__":
    logger.info("Starting web scraping with metadata")
    try:
        scrape_websites()
        logger.info("Web scraping completed successfully")
    except Exception as e:
        logger.exception("Error during web scraping")
        raise
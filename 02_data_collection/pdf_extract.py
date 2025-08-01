# Updated 02_data_collection/pdf_extract.py with metadata, layout info, and timezone-aware datetime
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import pdfplumber
from sys import path

# Load config
path.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

# ─── Logging setup ─────────────────────────────────────────────────────────────
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "pdf_extract.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def extract_pdfs():
    config = load_config()
    out_dir = Path(config["output"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use timezone-aware UTC timestamp
    crawl_date = datetime.now(timezone.utc).isoformat()
    for pdf_path in config["data_sources"]["pdfs"]:
        pdf_file = Path(pdf_path)
        logger.info(f"Reading {pdf_file.name}")
        pages = []
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    # Split headings vs. paragraphs naively by newlines
                    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                    headings = [ln for ln in lines if ln.isupper() and len(ln.split()) < 10]
                    paragraphs = [ln for ln in lines if ln not in headings]
                    pages.append({
                        "page": i,
                        "layout": {"headings": headings, "paragraphs": paragraphs},
                        "word_count": len(text.split())
                    })
        except Exception:
            logger.exception(f"Failed to extract {pdf_file.name}")
            continue

        data = {"source": str(pdf_file), "crawl_date": crawl_date, "pages": pages}
        out_path = out_dir / f"{pdf_file.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved PDF data to {out_path}")


if __name__ == "__main__":
    logger.info("Starting PDF extraction with metadata and layout")
    try:
        extract_pdfs()
        logger.info("PDF extraction completed successfully")
    except Exception:
        logger.exception("Error during PDF extraction")
        raise
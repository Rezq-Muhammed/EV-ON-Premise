# 03_data_processing/clean.py

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv

# ─── Load environment & config ─────────────────────────────────────────────────
load_dotenv()  # reads .env for HF_TOKEN, etc.

from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

cfg = load_config()
HF_TOKEN = os.getenv("HF_TOKEN")

# ─── Prepare directories ────────────────────────────────────────────────────────
RAW_DIR   = Path(cfg["output"]["raw_dir"])
CLEAN_DIR = Path(cfg["output"]["clean_dir"])
LOG_DIR   = Path(cfg["output"]["logs_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_DIR / "clean.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    cfg["base_model"],
    token=HF_TOKEN
)

# ─── Text utilities ────────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)            # collapse whitespace
    s = re.sub(r"&[a-z]+;", "", s)        # strip HTML entities
    return s.strip()

def is_garbage(s: str) -> bool:
    words = s.split()
    if len(words) < 5:
        return True
    non_alpha = sum(1 for c in s if not c.isalnum() and not c.isspace())
    return (non_alpha / max(len(s), 1)) > 0.3

# ─── Main processing ────────────────────────────────────────────────────────────
def process_file(json_path: Path, seen: set, chunk_limit: int):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    crawl_date = data.get("crawl_date", datetime.now(timezone.utc).isoformat())
    source_stem = json_path.stem  # Use file stem to avoid nested dirs

    # Collect all text snippets
    snippets = []
    for entry in data.get("sections", []) + data.get("pages", []):
        if isinstance(entry, dict):
            if entry.get("text"):
                snippets.append(entry["text"])
            elif entry.get("layout", {}).get("paragraphs"):
                snippets.extend(entry["layout"]["paragraphs"])
        elif isinstance(entry, str):
            snippets.append(entry)

    # Clean, filter, dedupe
    clean_paras = []
    for txt in snippets:
        if not isinstance(txt, str):
            continue
        clean = normalize_text(txt)
        if not clean or is_garbage(clean):
            continue
        h = hash(clean)
        if h in seen:
            continue
        seen.add(h)
        clean_paras.append(clean)

    if not clean_paras:
        logger.warning(f"No valid paragraphs in {json_path.name}")
        return

    # Group into chunks by token count
    chunks = []
    current_chunk = []
    current_tok_count = 0

    for para in clean_paras:
        toks = tokenizer.encode(para, add_special_tokens=False)
        if current_tok_count + len(toks) > chunk_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_tok_count = len(toks)
        else:
            current_chunk.append(para)
            current_tok_count += len(toks)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Write out each chunk
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{source_stem}_chunk_{idx}"
        out = {
            "source": data.get("source", source_stem),
            "crawl_date": crawl_date,
            "chunk_id": chunk_id,
            "text": chunk_text
        }
        out_path = CLEAN_DIR / f"{chunk_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"{json_path.name}: {len(clean_paras)} paras -> {len(chunks)} chunks")

def main():
    max_tokens = cfg["generation"]["max_tokens"]
    logger.info("=== Data Processing Started ===")
    seen_hashes = set()

    for file in tqdm(RAW_DIR.glob("*.json"), desc="Cleaning"):
        try:
            process_file(file, seen_hashes, max_tokens)
        except Exception:
            logger.exception(f"Error processing {file.name}")

    logger.info("=== Data Processing Completed ===")

if __name__ == "__main__":
    main()

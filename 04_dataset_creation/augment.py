# 04_dataset_creation/augment.py

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

# ─── Load config & env ─────────────────────────────────────────────────────────
load_dotenv()  # reads .env

from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

cfg = load_config()

# ─── Azure OpenAI client setup ────────────────────────────────────────────────
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME   = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # your chat deployment

if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, DEPLOYMENT_NAME]):
    raise ValueError(
        "Missing one of AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT in .env"
    )

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

# ─── Paths & logging ───────────────────────────────────────────────────────────
RAW_CLEAN_DIR = Path(cfg["output"]["clean_dir"])
DATASET_DIR   = Path(cfg["output"]["dataset_dir"])
LOG_DIR       = Path(cfg["output"]["logs_dir"])

for d in (DATASET_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "augment.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Generation parameters ─────────────────────────────────────────────────────
gen_cfg     = cfg.get("generation", {})
MAX_TOKENS  = gen_cfg.get("max_tokens", 512)
TEMPERATURE = gen_cfg.get("temperature", 0.7)

# ─── Prompt template ────────────────────────────────────────────────────────────
PROMPT = (
    "You are an expert on {domain}.\n"
    "Given the following text excerpt, generate exactly one clear question and its answer.\n\n"
    "Text:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
    "Respond as JSON with keys 'question' and 'answer'."
)

def generate_qa(chunk: str) -> dict:
    messages = [
        {"role": "system", "content": f"Domain: {cfg['domain']}"},
        {"role": "user",   "content": PROMPT.format(domain=cfg["domain"], chunk=chunk)}
    ]
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    content = resp.choices[0].message.content.strip()
    return json.loads(content)

def main():
    logger.info("=== Azure Dataset Augmentation Started ===")
    examples = []

    for file in tqdm(RAW_CLEAN_DIR.glob("*.json"), desc="Chunks"):
        data = json.loads(file.read_text(encoding="utf-8"))
        chunk = data.get("text", "").strip()
        if not chunk:
            logger.warning(f"No text in {file.name}, skipping")
            continue

        try:
            qa = generate_qa(chunk)
            examples.append({"prompt": qa["question"], "completion": qa["answer"]})
        except Exception as e:
            logger.exception(f"Failed on {file.name}: {e}")

    if not examples:
        logger.error("No examples generated; aborting")
        return

    # ─── Split and write JSONL ─────────────────────────────────────────────────
    split = int(len(examples) * 0.9)
    train, valid = examples[:split], examples[split:]

    with open(DATASET_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(DATASET_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(train)} train and {len(valid)} valid examples")
    logger.info("=== Azure Dataset Augmentation Completed ===")

if __name__ == "__main__":
    main()

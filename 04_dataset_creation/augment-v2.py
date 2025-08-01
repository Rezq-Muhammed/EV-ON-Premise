# 04_dataset_creation/augment.py

import os
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from dotenv import load_dotenv

# ─── Load config & env ─────────────────────────────────────────────────────────
load_dotenv()
from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

HF_TOKEN = os.getenv("HF_TOKEN")
cfg       = load_config()
gen_cfg   = cfg.get("generation", {})
HF_QA_MODEL  = gen_cfg.get("qa_model", "Qwen/Qwen2.5-7B-Instruct")
MAX_TOKENS = gen_cfg.get("max_tokens", 512)
TEMPERATURE= gen_cfg.get("temperature", 0.7)

data_dir  = Path(cfg["output"]["clean_dir"])
out_dir   = Path(cfg["output"]["dataset_dir"])
log_dir   = Path(cfg["output"]["logs_dir"])
for d in (out_dir, log_dir): d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_dir / "augment.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load model & tokenizer ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(HF_QA_MODEL, use_auth_token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(HF_QA_MODEL, token=HF_TOKEN, trust_remote_code=True)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# ─── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an AI assistant that extracts a question-answer pair from the text below.

Text:
{text}

Respond in JSON format:
{{
  "question": "Your question here?",
  "answer": "The corresponding answer here."
}}
"""

# ─── Generate QA ───────────────────────────────────────────────────────────────
def generate_qa(text):
    prompt = PROMPT_TEMPLATE.format(text=text.strip())
    response = qa_pipeline(prompt, max_new_tokens=MAX_TOKENS, do_sample=False)[0]["generated_text"]
    json_part = response[len(prompt):].strip()
    print(json_part)

    try:
        return json.loads(json_part)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON: {json_part}")
        return None

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Dataset Augmentation Started ===")
    examples = []

    for f in data_dir.glob("*.json"):
        chunk = json.loads(f.read_text(encoding="utf-8")).get("text", "").strip()
        if not chunk:
            logger.warning(f"Empty chunk {f.name}, skipping.")
            continue
        try:
            qa = generate_qa(chunk)
            examples.append({"prompt": qa["question"], "completion": qa["answer"]})
        except Exception as e:
            logger.error(f"Skipping {f.name}: {e}")

    if not examples:
        logger.error("No examples generated; aborting.")
        return

    split = int(0.9 * len(examples))
    train, valid = examples[:split], examples[split:]

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as tf:
        for ex in train:
            tf.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_dir / "valid.jsonl", "w", encoding="utf-8") as vf:
        for ex in valid:
            vf.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(train)} train and {len(valid)} valid examples")
    logger.info("=== Dataset Augmentation Completed ===")

if __name__ == "__main__":
    main()

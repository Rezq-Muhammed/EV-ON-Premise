# 06_evaluation/run_eval.py

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel
import evaluate

# ─── Load config & env ─────────────────────────────────────────────────────────
load_dotenv()
from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

cfg = load_config()
BASE_MODEL     = cfg["base_model"]
PEFT_MODEL_DIR = Path(cfg["output"]["models_dir"]) / "final"
DATA_DIR       = Path(cfg["output"]["dataset_dir"])
LOG_DIR        = Path(cfg["output"]["logs_dir"])

# ensure log dir exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_DIR / "eval.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load tokenizer and fine-tuned model ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": "cpu"})
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_DIR)
model.eval()

# ─── Prepare QA pipeline (CPU-only) ─────────────────────────────────────────────
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=cfg["generation"]["max_tokens"],
    temperature=cfg["generation"]["temperature"]
)

# ─── Load validation set ─────────────────────────────────────────────────────────
data_files = {"validation": str(DATA_DIR / "valid.jsonl")}
dataset = load_dataset("json", data_files=data_files)["validation"]

# ─── Metric setup using SQuAD-style EM & F1 ───────────────────────────────────
squad_metric = evaluate.load("squad")

# ─── Evaluation loop ────────────────────────────────────────────────────────────
predictions, references = [], []
for idx, ex in enumerate(tqdm(dataset, desc="Evaluating")):
    prompt = ex["prompt"]
    target = ex["completion"].strip()

    output = qa_pipeline(prompt)[0]["generated_text"].replace(prompt, "").strip()
    predictions.append({"id": str(idx), "prediction_text": output})
    references.append({
        "id": str(idx),
        "answers": {"text": [target], "answer_start": [0]}
    })

# ─── Compute metrics ────────────────────────────────────────────────────────────
results = squad_metric.compute(predictions=predictions, references=references)

# ─── Log results ───────────────────────────────────────────────────────────────
exact = results.get("exact") or results.get("exact_match")
f1    = results.get("f1") or results.get("f1_score")
logger.info("Evaluation Results:")
logger.info(f"Exact Match: {exact}")
logger.info(f"F1 Score: {f1}")
print(f"Exact Match: {exact}")
print(f"F1 Score: {f1}")

print("Evaluation complete. See eval.log for full details.")

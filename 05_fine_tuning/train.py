# 05_fine_tuning/train.py

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# ─── Load config & env ─────────────────────────────────────────────────────────
load_dotenv()
from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config

cfg        = load_config()
BASE_MODEL = cfg["base_model"]            # e.g. "EleutherAI/gpt-neo-1.3B"
OUT_DIR    = Path(cfg["output"]["models_dir"])
LOG_DIR    = Path(cfg["output"]["logs_dir"])
DATA_DIR   = Path(cfg["output"]["dataset_dir"])
MAX_TOKENS = cfg["generation"]["max_tokens"]

# Ensure dirs exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_DIR / "train.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Tokenizer & Model ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model entirely on CPU
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "cpu"},
    torch_dtype="auto"
)

# ─── Apply LoRA Adapter ─────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_cfg)

# ─── Prepare Dataset ───────────────────────────────────────────────────────────
data_files = {
    "train": str(DATA_DIR / "train.jsonl"),
    "validation": str(DATA_DIR / "valid.jsonl")
}
dataset = load_dataset("json", data_files=data_files)

# Tokenization function with padding + truncation
def tokenize_fn(examples):
    inputs = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
    enc = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKENS
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ─── Data collator ──────────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ─── Training Arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_dir=str(LOG_DIR),
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    fp16=False,            # disable mixed precision on CPU
    do_train=True,
    do_eval=True
)

# ─── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    processing_class=tokenizer
)

# ─── Run training ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ""

        def write(self, message):
            if message.strip() != "":
                self.logger.log(self.level, message.strip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    logger.info("Starting CPU-only fine-tuning")
    trainer.train()
    trainer.save_model(OUT_DIR / "final")
    logger.info("Fine-tuning complete")

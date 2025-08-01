# 07_deployment/app.py

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import PeftModel

# ─── Load environment & config ─────────────────────────────────────────────────
load_dotenv()  # loads variables from .env
from sys import path as _p
_p.append(str(Path(__file__).parents[1] / "01_config"))
from load_config import load_config
cfg = load_config()

# ─── Read settings ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("FASTAPI_API_KEY", "my_fastapi_key")  # set in .env
BASE_MODEL = cfg["base_model"]
PEFT_MODEL_DIR = Path(cfg["output"]["models_dir"]) / "final"
MAX_TOKENS = cfg["generation"]["max_tokens"]
TEMPERATURE = cfg["generation"]["temperature"]
LOG_DIR = Path(cfg["output"]["logs_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_DIR / "inference.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("deployment")

# ─── Security ──────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-KEY")

def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access with key: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# ─── Load tokenizer & LoRA-adapted model ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "cpu"}
)
model = PeftModel.from_pretrained(base, PEFT_MODEL_DIR)
model.eval()

# ─── Create inference pipeline ──────────────────────────────────────────────────
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_TOKENS,
    temperature=TEMPERATURE
)

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="EV Charging QA API",
    description="Question-answering on EV charging stations",
    version="1.0.0"
)

class QARequest(BaseModel):
    prompt: str

class QAResponse(BaseModel):
    answer: str

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "OK"}

@app.post("/predict", response_model=QAResponse, tags=["inference"])
async def predict(request: QARequest, api_key: str = Depends(validate_api_key)):
    logger.info(f"Prompt received: {request.prompt}")
    try:
        out = generator(request.prompt)[0]["generated_text"].strip()
        if out.startswith(request.prompt):
            out = out[len(request.prompt):].strip()
        logger.info(f"Response: {out}")
        return QAResponse(answer=out)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Use /predict with X-API-KEY header."}

# To run:
# uvicorn 07_deployment.app:app --host 0.0.0.0 --port 8000

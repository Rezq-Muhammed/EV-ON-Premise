# Fine‑Tune Small LLM Pipeline

This repository provides a complete, modular pipeline for fine-tuning small language models (≤7B parameters) using domain-specific data, with support for evaluation and deployment as an API.

---

## 📁 Project Structure

```
project_root/
├── 01_config/               # Configuration files and environment loader
├── 02_data_collection/      # Scraping and document extraction
├── 03_data_processing/      # Cleaning, deduplication, and preprocessing
├── 04_dataset_creation/     # Dataset augmentation and formatting
├── 05_fine_tuning/          # LoRA fine-tuning scripts
├── 06_evaluation/           # Evaluation scripts (SQuAD metrics, F1, EM)
├── 07_deployment/           # FastAPI app for model inference
├── data/                    # Input data (PDFs, HTML, etc.)
├── output/                  # Artifacts (datasets, models, logs)
│   ├── dataset/
│   ├── logs/
│   ├── models/
│   └── raw/clean/
├── .env                     # Main environment variables (ignored by Git)
├── example.env              # Template for environment setup
├── pip_requirements.txt     # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Rezq-Muhammed/EV-ON-Premise.git
cd EV-ON-Premise
```

### 2. Create and activate a conda environment

```bash
conda create -n ev_env python=3.10
conda activate ev_env
```

### 3. Install required packages

```bash
pip install -r pip_requirements.txt
```

### 4. Configure environment variables


Edit `.env`:

```ini
OPENAI_API_KEY=your_openai_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_DEPLOYMENT=your_openai_model_name_here
AZURE_OPENAI_API_VERSION=your_openai_api_version_here
HF_TOKEN=your_huggingface_token_here                        # stored securely in .env
FASTAPI_API_KEY=your_fastapi_key_here
```

---

## 🚀 Pipeline Steps

### Step 1: Data Collection

```bash
python 02_data_collection/scrape.py
python 02_data_collection/pdf_extract.py
```

### Step 2: Data Processing

```bash
python 03_data_processing/clean.py
```

### Step 3: Dataset Creation

```bash
python 04_dataset_creation/augment.py        # Generating dataset using AzureOpenAI
python 04_dataset_creation/augment-v2.py     # An additional and alternative way to generate the dataset using an on-premise technique such as Qwen/Qwen2.5-7B-Instruct
```

### Step 4: Fine‑Tuning

```bash
python 05_fine_tuning/train.py
```

### Step 5: Evaluation

```bash
python 06_evaluation/run_eval.py
```

### Step 6: Deployment via FastAPI

```bash
uvicorn 07_deployment.app:app --host 0.0.0.0 --port 8000
```

---

## 🔐 Authentication & API Key

The FastAPI service uses an authentication key stored in `.env` under:

```env
FASTAPI_API_KEY=your_fastapi_key_here
```

This key must be included in the `x-api-key` header when making prediction requests.

---

## 💬 Prediction via cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: 17uG4h8Kq9ZsX3eLp5RnYt2Vc1" \
  -d '{"prompt": "What is the impact of EV infrastructure on urban areas?"}'
```

---

## ☁️ GitHub Deployment

1. Initialize a Git repository:

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Push to GitHub:

```bash
git remote add origin https://github.com/yourusername/llm-finetune-pipeline.git
git push -u origin main
```

---

## ✅ Notes

- Logs are saved under `output/logs/`
- All configs are loaded via `.env` and `01_config/config.yaml`
- Compatible with `Qwen/Qwen2.5-7B-Instruct` on CPU

---

## 🛠️ License

MIT License

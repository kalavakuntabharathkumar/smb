# LLM Evaluation Platform

A platform for the full LLM lifecycle: supervised fine-tuning with LoRA →
DPO alignment → post-training quantisation → multi-backend serving benchmark →
evaluation. The FastAPI backend and Streamlit dashboard run on a standard
Linux machine or in Docker; GPU-intensive training stages (SFT, DPO,
quantisation) ran on a free-tier Colab/Kaggle GPU and the resulting
checkpoints are loaded here for serving and evaluation.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Fine-tuning | PyTorch + PEFT (LoRA) |
| Alignment | trl DPOTrainer |
| Quantisation | AutoAWQ / AutoGPTQ |
| Serving | vLLM · HuggingFace TGI · llama.cpp |
| Experiment tracking | Weights & Biases |
| Evaluation | DistilBERT classifier · scikit-learn · scipy |

---

## Pipeline overview

```
Dataset
  │
  ▼
LoRA SFT  ──────────────────────────────────────► W&B (project: llm-sft)
  │   training/lora_finetune.py
  │   training/train_loop.py  (raw PyTorch loop)
  ▼
DPO alignment ─────────────────────────────────► W&B (project: llm-dpo)
  │   training/dpo_train.py
  ▼
Quantisation
  │   training/quantize.py
  ▼
Serving benchmark
  │   training/serving_benchmark.py
  ▼
Evaluation pipeline
  │   evaluation/pipeline.py
  ▼
FastAPI + Streamlit dashboard
```

## Project structure

- `backend/` FastAPI API and LLM service layer
- `evaluation/` feature engineering, metrics, statistics, tuning, and evaluation pipeline
- `training/` LoRA, DPO, quantisation, and serving benchmark utilities
- `datasets/` evaluation and safety datasets
- `frontend/` Streamlit dashboard
- `tests/` basic and pipeline tests
- `configs/` runtime configuration

## Running locally

Install dependencies and run the API:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Run the dashboard separately:

```bash
streamlit run frontend/app.py
```

Docker is also supported through `docker-compose.yml`.

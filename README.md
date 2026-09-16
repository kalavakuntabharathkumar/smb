# LLM Evaluation Platform

**Project duration:** July 27, 2026 – August 12, 2026

A platform for the full LLM lifecycle: supervised fine-tuning with LoRA → DPO alignment → post-training quantisation → multi-backend serving benchmark → evaluation. The FastAPI backend and Streamlit dashboard run on Linux or Docker; GPU-intensive training stages can be run on a Colab/Kaggle GPU.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Fine-tuning | PyTorch + PEFT (LoRA) |
| Alignment | TRL DPOTrainer |
| Quantisation | AutoAWQ / AutoGPTQ |
| Serving | vLLM · HuggingFace TGI · llama.cpp |
| Evaluation | scikit-learn · scipy · pandas |

## Pipeline

```text
Dataset → LoRA SFT → DPO alignment → Quantisation → Serving benchmark → Evaluation
                                      ↓
                               FastAPI + Streamlit
```

## Structure

- `backend/` — FastAPI routes and LLM service layer
- `evaluation/` — feature engineering, metrics and statistical analysis
- `training/` — LoRA, DPO, quantisation and serving benchmark utilities
- `datasets/` — evaluation and safety datasets
- `frontend/` — Streamlit dashboard
- `tests/` — automated smoke tests

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

In another terminal:

```bash
streamlit run frontend/app.py
```

Docker support is provided through `Dockerfile` and `docker-compose.yml`.

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")
st.title("LLM Evaluation Platform")

tab_eval, tab_run = st.tabs(["Evaluation", "Experiments"])

with tab_eval:
    st.header("Model Evaluation")
    output = st.text_area("Model output", height=150)
    ground_truth = st.text_area("Ground truth (optional)", height=100)
    human_score = st.slider("Human score", 0.0, 1.0, 0.8)
    if st.button("Run ML evaluation"):
        try:
            response = requests.post(f"{API_BASE}/evaluate-report/", json={"model_output": output, "ground_truth": ground_truth, "human_score": human_score}, timeout=60)
            st.json(response.json())
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with tab_run:
    st.header("Offline experiment pipeline")
    if st.button("Run experiment"):
        try:
            response = requests.get(f"{API_BASE}/experiments/run/", timeout=120)
            st.json(response.json())
        except Exception as exc:
            st.error(f"Request failed: {exc}")

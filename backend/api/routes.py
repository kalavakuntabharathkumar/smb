from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.evaluator import evaluate_text
from evaluation.pipeline import run_offline_evaluation, run_single_report

router = APIRouter()

class EvaluationRequest(BaseModel):
    model_output: str
    ground_truth: str = ""
    human_score: float | None = None

@router.get("/evaluate")
def evaluate(prompt: str, response: str):
    return evaluate_text(prompt, response)

@router.post("/evaluate-report/")
def evaluate_report(request: EvaluationRequest):
    return run_single_report(request.model_output, request.ground_truth, request.human_score)

@router.get("/experiments/run/")
def run_experiments():
    return run_offline_evaluation()

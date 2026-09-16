from fastapi import APIRouter
from backend.services.evaluator import evaluate_text

router = APIRouter()

@router.get("/evaluate")
def evaluate(prompt: str, response: str):
    return evaluate_text(prompt, response)

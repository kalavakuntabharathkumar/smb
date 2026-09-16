from fastapi import FastAPI
from backend.api.routes import router

app = FastAPI(title="LLM Evaluation Platform", version="1.0.0")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}

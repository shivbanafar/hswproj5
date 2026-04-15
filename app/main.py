from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model as model_module
from app.routers import sentiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_module.analyzer = model_module.VADERAnalyzer()
    yield
    model_module.analyzer = None


app = FastAPI(
    title="Financial Sentiment Analyzer",
    description=(
        "Analyze financial news headlines using **VADER** — "
        "a rule-based NLP model for sentiment analysis. Returns positive, negative, "
        "or neutral sentiment with per-class confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "model": model_module.MODEL_NAME}

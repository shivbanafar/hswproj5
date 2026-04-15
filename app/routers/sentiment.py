from fastapi import APIRouter, HTTPException, Query

from app import model
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HeadlineRequest,
    SamplesResponse,
    SentimentResult,
)

router = APIRouter()


@router.post("/analyze", response_model=SentimentResult, summary="Analyze a single headline")
async def analyze_headline(request: HeadlineRequest):
    """
    Analyze the sentiment of a single financial news headline.

    Returns **positive**, **negative**, or **neutral** with a confidence score
    and per-class softmax probabilities.
    """
    if not request.headline.strip():
        raise HTTPException(status_code=422, detail="Headline cannot be empty.")
    return model.analyzer.analyze(request.headline)


@router.get("/analyze", response_model=SentimentResult, summary="Analyze a headline (GET)")
async def analyze_headline_get(
    headline: str = Query(..., min_length=1, description="Financial news headline to analyze")
):
    """
    Convenience GET endpoint — useful for quick browser or curl testing.
    """
    return model.analyzer.analyze(headline)


@router.post("/analyze/batch", response_model=BatchResponse, summary="Analyze multiple headlines")
async def analyze_batch(request: BatchRequest):
    """
    Analyze up to **50 headlines** in a single request.
    """
    if not request.headlines:
        raise HTTPException(status_code=422, detail="Headlines list cannot be empty.")
    if len(request.headlines) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 headlines per batch request.")
    results = model.analyzer.analyze_batch(request.headlines)
    return BatchResponse(results=results, model_used=model.MODEL_NAME)


@router.get("/samples", response_model=SamplesResponse, summary="Get labeled example headlines")
async def get_samples(
    count: int = Query(10, ge=1, le=100, description="Number of samples to return"),
    sentiment: str | None = Query(
        None,
        description="Filter by sentiment: positive, negative, or neutral",
        pattern="^(positive|negative|neutral)$",
    ),
):
    """
    Return labeled example headlines from the **Financial PhraseBank** dataset
    (sentences with 100% annotator agreement).

    Use these to pre-populate the dashboard or verify model behavior.
    """
    samples = model.analyzer.get_samples(count=count, sentiment=sentiment)
    return SamplesResponse(
        samples=samples,
        dataset="takala/financial_phrasebank (sentences_allagree)",
        total_available=model.analyzer.sample_count,
    )

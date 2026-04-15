from typing import Literal

from pydantic import BaseModel, ConfigDict


class HeadlineRequest(BaseModel):
    headline: str
    model_config = ConfigDict(json_schema_extra={
        "example": {"headline": "Apple beats Q3 earnings expectations by 12%"}
    })


class SentimentResult(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    scores: dict[str, float]


class BatchRequest(BaseModel):
    headlines: list[str]
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "headlines": [
                "Apple beats Q3 earnings expectations by 12%",
                "Company files for Chapter 11 bankruptcy",
                "Board approves annual dividend payment"
            ]
        }
    })


class BatchResponse(BaseModel):
    results: list[SentimentResult]
    model_used: str


class SampleHeadline(BaseModel):
    sentence: str
    label: Literal["positive", "negative", "neutral"]


class SamplesResponse(BaseModel):
    samples: list[SampleHeadline]
    dataset: str
    total_available: int

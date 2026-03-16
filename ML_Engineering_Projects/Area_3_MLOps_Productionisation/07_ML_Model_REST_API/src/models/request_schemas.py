from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to classify (max 32)",
    )

    @field_validator("texts")
    @classmethod
    def strip_texts(cls, v: list[str]) -> list[str]:
        return [t.strip() for t in v if t.strip()]

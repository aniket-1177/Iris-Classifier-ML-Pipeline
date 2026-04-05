"""
Pydantic request/response schemas for the FastAPI application.
Provides automatic validation, serialization, and OpenAPI documentation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Single-sample inference request."""

    sepal_length: float = Field(
        ..., ge=0.0, le=20.0, description="Sepal length in cm", example=5.1
    )
    sepal_width: float = Field(
        ..., ge=0.0, le=20.0, description="Sepal width in cm", example=3.5
    )
    petal_length: float = Field(
        ..., ge=0.0, le=20.0, description="Petal length in cm", example=1.4
    )
    petal_width: float = Field(
        ..., ge=0.0, le=20.0, description="Petal width in cm", example=0.2
    )

    def to_feature_list(self) -> List[float]:
        return [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]


class PredictResponse(BaseModel):
    """Single-sample inference response."""

    predicted_class: str = Field(..., description="Predicted Iris species")
    confidence: float = Field(..., description="Confidence score of the top prediction")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Probability per class"
    )


class BatchPredictRequest(BaseModel):
    """Batch inference request — up to 100 samples."""

    samples: List[PredictRequest] = Field(
        ..., min_length=1, max_length=100, description="List of samples to predict"
    )


class BatchPredictResponse(BaseModel):
    """Batch inference response."""

    predictions: List[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    model_classes: Optional[List[str]] = None
    version: str

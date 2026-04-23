"""
Prediction endpoints router.
Keeps routing logic separate from the main app entrypoint.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.inference.predictor import ModelNotFoundError, Predictor, get_predictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="Single-sample prediction",
    description="Submit one flower's measurements and get the predicted Iris species.",
)
def predict(
    request: PredictRequest,
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    try:
        result = predictor.predict(request.to_feature_list())
        logger.info("Prediction: %s (confidence=%.4f)", result["predicted_class"], result["confidence"])
        return PredictResponse(**result)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch prediction",
    description="Submit multiple samples in one request.",
)
def predict_batch(
    request: BatchPredictRequest,
    predictor: Predictor = Depends(get_predictor),
) -> BatchPredictResponse:
    try:
        predictions = [
            PredictResponse(**predictor.predict(sample.to_feature_list()))
            for sample in request.samples
        ]
        logger.info("Batch prediction | n=%d", len(predictions))
        return BatchPredictResponse(predictions=predictions, total=len(predictions))
    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

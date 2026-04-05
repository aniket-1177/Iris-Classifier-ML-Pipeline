"""
FastAPI application factory.
Registers routers, startup/shutdown events, and global error handlers.
"""

import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers.predict import router as predict_router
from api.schemas import HealthResponse
from src.config import API_TITLE, API_VERSION, LOG_FORMAT, LOG_LEVEL
from src.inference.predictor import ModelNotFoundError, get_predictor

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "REST API for serving the Iris species classifier. "
            "Train the model first with `python scripts/train.py`."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(predict_router)

    # ── Lifecycle ─────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Starting %s v%s", API_TITLE, API_VERSION)
        try:
            predictor = get_predictor()
            logger.info(
                "Model ready | classes=%s", predictor.model_classes
            )
        except ModelNotFoundError as e:
            logger.warning("⚠️  Model not loaded: %s", e)

    # ── Health ────────────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
    )
    def health() -> HealthResponse:
        try:
            predictor = get_predictor()
            return HealthResponse(
                status="ok",
                model_loaded=True,
                model_classes=predictor.model_classes,
                version=API_VERSION,
            )
        except ModelNotFoundError:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                version=API_VERSION,
            )

    @app.get("/", tags=["System"], include_in_schema=False)
    def root():
        return JSONResponse(
            {"message": f"Welcome to {API_TITLE}. Visit /docs for the API reference."}
        )

    return app


app = create_app()

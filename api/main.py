"""
FastAPI Backend — Laptop Price Prediction API
Endpoints: POST /predict  |  GET /health  |  GET /models
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict as _predict


#**********************************************************
#                   Logging Setup
#**********************************************************
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_file_handler = RotatingFileHandler(
    _LOG_DIR / "api.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(_file_handler)
    log.info("File logging aktiv → %s", _LOG_DIR / "api.log")

    required = ["production_model.pkl", "feature_engineer.pkl",
                "quantile_low.pkl", "quantile_high.pkl"]
    missing = [f for f in required if not (PROCESSED_DIR / f).exists()]
    if missing:
        log.error("Missing model artifacts: %s — run 'make train' first.", missing)
        raise RuntimeError(
            f"Model artifacts not found: {missing}. Run 'make train' before starting the API."
        )

    yield


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


#**********************************************************
#                   FastAPI
#**********************************************************
app = FastAPI(
    title="Laptop Price Prediction API",
    description="Preisvorhersage für Laptops mit MLflow + SHAP Explainability",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────

class LaptopSpecs(BaseModel):
    brand:        str  = Field(...,  example="Lenovo")
    cpu:          str  = Field(...,  example="Intel Core i7-13700H")
    gpu:          str  = Field("",   example="NVIDIA RTX 4060")
    ram_gb:       int  = Field(...,  example=16)
    storage_gb:   int  = Field(...,  example=512)
    storage_type: str  = Field("SSD", example="SSD")
    display_inch: float= Field(...,  example=15.6)
    resolution:   str  = Field("",   example="1920x1080")
    os:           str  = Field("Windows 11", example="Windows 11")
    weight_kg:    float= Field(0.0,  example=1.8)


class PredictionResponse(BaseModel):
    predicted_price:     float
    confidence_interval: list[float]
    shap_values:         dict[str, float]
    base_value:          float
    shap_plot_base64:    str
    feature_values:      dict[str, float]


class ModelInfo(BaseModel):
    run_id:    str
    run_name:  str
    metrics:   dict[str, float]
    status:    str

#**********************************************************
#                   Endpoints
#**********************************************************
@app.get("/health")
def health():
    return {"status": "ok", "model": "LaptopPriceModel@Production"}


@app.post("/predict", response_model=PredictionResponse)
def predict(specs: LaptopSpecs):
    try:
        result = _predict(specs.model_dump())
        return result
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/shap/summary")
def shap_summary():
    """Gibt den SHAP Summary-Plot und Bar-Plot als base64 zurück."""
    summary_path = PROCESSED_DIR / "shap_summary.png"
    bar_path     = PROCESSED_DIR / "shap_bar.png"

    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="SHAP-Plots nicht gefunden. Training ausführen.")

    return {
        "summary_b64": base64.b64encode(summary_path.read_bytes()).decode(),
        "bar_b64":     base64.b64encode(bar_path.read_bytes()).decode() if bar_path.exists() else "",
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("laptop_price_prediction")
        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=10,
        )
        return [
            ModelInfo(
                run_id=r.info.run_id,
                run_name=r.info.run_name or r.info.run_id[:8],
                metrics={k: round(v, 4) for k, v in r.data.metrics.items()},
                status=r.info.status,
            )
            for r in runs
        ]
    except Exception as exc:
        log.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(exc))

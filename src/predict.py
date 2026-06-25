"""
Prediction-Modul: lädt das beste Modell aus MLflow und stellt predict() bereit.
"""

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import LaptopFeatureEngineer, PROCESSED_DIR
from src.shap_analysis import SHAPAnalyzer

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Lädt production_model.pkl + feature_engineer.pkl (cached)."""
    model = joblib.load(PROCESSED_DIR / "production_model.pkl")
    eng   = joblib.load(PROCESSED_DIR / "feature_engineer.pkl")
    log.info("Loaded production model: %s", type(model).__name__)
    return model, eng


@lru_cache(maxsize=1)
def _load_quantile_models():
    """Lädt 10%/90%-Quantil-Modelle (cached)."""
    q_low  = joblib.load(PROCESSED_DIR / "quantile_low.pkl")
    q_high = joblib.load(PROCESSED_DIR / "quantile_high.pkl")
    return q_low, q_high


def predict(specs: dict) -> dict:
    """
    specs: dict with raw laptop fields (brand, cpu, gpu, ram_gb, ...)
    Returns: {predicted_price, shap_values, shap_plot_base64, feature_values}
    """
    model, eng = _load_model()
    df = pd.DataFrame([specs])
    X  = eng.transform(df)

    price = float(model.predict(X)[0])

    # Echtes Konfidenzintervall via Quantile Regression
    q_low, q_high = _load_quantile_models()
    ci_low  = round(float(q_low.predict(X)[0]),  2)
    ci_high = round(float(q_high.predict(X)[0]), 2)

    # SHAP — use small background (zeros) for speed in API context
    background = np.zeros((1, X.shape[1]))
    analyzer   = SHAPAnalyzer(model, eng.feature_names_, background)
    shap_result = analyzer.explain_single(X[0])
    waterfall   = analyzer.waterfall_plot_base64(X[0])

    return {
        "predicted_price":    round(price, 2),
        "confidence_interval": [ci_low, ci_high],
        "shap_values":        shap_result["shap_values"],
        "base_value":         shap_result["base_value"],
        "shap_plot_base64":   waterfall,
        "feature_values":     {
            name: round(float(val), 4)
            for name, val in zip(eng.feature_names_, X[0])
        },
    }

"""
SHAP Explainability — Waterfall- und Summary-Plots für Laptop-Preisvorhersagen.
"""

import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

log = logging.getLogger(__name__)


class SHAPAnalyzer:
    def __init__(self, model, feature_names: list[str], X_background: np.ndarray):
        self.model        = model
        self.feature_names = feature_names

        # Choose explainer based on model type
        model_type = type(model).__name__
        if model_type in ("LinearRegression", "Ridge", "Lasso"):
            self.explainer = shap.LinearExplainer(model, X_background)
            self._tree_based = False
        else:
            self.explainer = shap.TreeExplainer(model)
            self._tree_based = True

        log.info("SHAPAnalyzer ready (%s explainer)", "Tree" if self._tree_based else "Linear")

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain one prediction.
        x: 1D or 2D array (1, n_features)
        Returns: {"base_value": float, "shap_values": {feature: value}, "prediction": float}
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        sv = self.explainer(x)
        shap_values = sv.values[0]
        base_value  = float(sv.base_values[0]) if sv.base_values.ndim > 0 else float(sv.base_values)

        return {
            "base_value":  base_value,
            "prediction":  base_value + float(shap_values.sum()),
            "shap_values": {
                name: round(float(val), 4)
                for name, val in zip(self.feature_names, shap_values)
            },
        }

    def waterfall_plot_base64(self, x: np.ndarray, max_display: int = 12) -> str:
        """
        Generate a SHAP waterfall plot for one sample.
        Returns a base64-encoded PNG string suitable for HTML/Streamlit embedding.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        sv = self.explainer(x)
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv[0], max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close("all")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def summary_plot_base64(self, X: np.ndarray, max_display: int = 15) -> str:
        """
        Beeswarm summary plot over the full dataset.
        Returns base64-encoded PNG.
        """
        sv = self.explainer(X)
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv.values, X, feature_names=self.feature_names,
                          max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close("all")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def bar_plot_base64(self, X: np.ndarray, max_display: int = 15) -> str:
        """Mean absolute SHAP bar chart — global feature importance."""
        sv = self.explainer(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(sv, max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close("all")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

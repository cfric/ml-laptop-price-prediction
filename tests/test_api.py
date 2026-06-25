import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)

MOCK_PREDICT_RESULT = {
    "predicted_price": 1250.0,
    "confidence_interval": [1100.0, 1400.0],
    "shap_values": {"cpu_score": 200.0, "gpu_score": 50.0},
    "base_value": 1000.0,
    "shap_plot_base64": base64.b64encode(b"fake-png").decode(),
    "feature_values": {"cpu_score": 7.0, "gpu_score": 7.5},
}

VALID_PAYLOAD = {
    "brand": "Dell",
    "cpu": "Intel Core i7-13700H",
    "gpu": "NVIDIA RTX 4060",
    "ram_gb": 16,
    "storage_gb": 512,
    "storage_type": "SSD",
    "display_inch": 15.6,
    "resolution": "1920x1080",
    "os": "Windows 11",
    "weight_kg": 1.8,
}


class TestHealth:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_contains_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestPredict:
    @patch("api.main._predict", return_value=MOCK_PREDICT_RESULT)
    def test_valid_request_returns_200(self, mock_predict):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    @patch("api.main._predict", return_value=MOCK_PREDICT_RESULT)
    def test_response_contains_price_and_ci(self, mock_predict):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "predicted_price" in data
        assert "confidence_interval" in data
        assert len(data["confidence_interval"]) == 2

    @patch("api.main._predict", return_value=MOCK_PREDICT_RESULT)
    def test_price_is_positive(self, mock_predict):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["predicted_price"] > 0

    def test_missing_required_field_returns_422(self):
        payload = VALID_PAYLOAD.copy()
        del payload["ram_gb"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_brand_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "brand"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    @patch("api.main._predict", side_effect=RuntimeError("Modell nicht geladen"))
    def test_internal_error_returns_500(self, mock_predict):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 500


class TestShapSummary:
    def test_missing_file_returns_404(self, tmp_path):
        with patch("api.main.PROCESSED_DIR", tmp_path):
            resp = client.get("/shap/summary")
        assert resp.status_code == 404

    def test_existing_files_returns_200(self, tmp_path):
        (tmp_path / "shap_summary.png").write_bytes(b"fake-summary-png")
        (tmp_path / "shap_bar.png").write_bytes(b"fake-bar-png")
        with patch("api.main.PROCESSED_DIR", tmp_path):
            resp = client.get("/shap/summary")
        assert resp.status_code == 200

    def test_response_contains_base64_data(self, tmp_path):
        (tmp_path / "shap_summary.png").write_bytes(b"fake-summary-png")
        (tmp_path / "shap_bar.png").write_bytes(b"fake-bar-png")
        with patch("api.main.PROCESSED_DIR", tmp_path):
            resp = client.get("/shap/summary")
        data = resp.json()
        assert len(data["summary_b64"]) > 0
        assert len(data["bar_b64"]) > 0

# Laptop Price Prediction — End-to-End ML Pipeline

[![CI](https://github.com/cfric/ml-laptop-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/cfric/ml-laptop-price-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.19-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

End-to-end ML project: collect laptop prices via web scraping, compare four regression models with MLflow, automatically register the best model under a Production alias, serve predictions via FastAPI, and visualize SHAP explanations in a Streamlit dashboard — fully containerized with Docker.

---

## Architecture

```
┌────────────────────────────────┐
│          Data Collection       │
│  nbb_scraper.py  (Playwright)  │
│  generate_dataset.py           │
└───────────────┬────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│              Feature Engineering               │
│  LaptopFeatureEngineer  (sklearn-compatible)  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │       MLflow Tracking      │
          │  LinearRegression           │
          │  RandomForest               │
          │  XGBoost  (Optuna-Tuned)   │
          │  LightGBM (Optuna-Tuned)   │
          └────────────┬───────────────┘
                       │  best model → @Production
                       ▼
     ┌─────────────────────────────────────────┐
     │           FastAPI Backend               │
     │  POST /predict · GET /models · /health  │
     │  SHAP Waterfall Plot (Base64)           │
     └─────────────────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Streamlit Frontend   │
              │   Price Forecast + CI  │
              │   SHAP Analysis        │
              └────────────────────────┘
```

---

## Prerequisites

- Python 3.12
- Docker & Docker Compose (for the container stack)
- Git
- `make` (on Windows: via [Git Bash](https://gitforwindows.org/) or WSL2)

---

## Quickstart

```bash
# 1 — Clone the repository
git clone https://github.com/cfric/ml-laptop-price-prediction.git
cd ml-laptop-price-prediction

# 2 — Set up environment (venv + Playwright browser)
make setup

# 3 — Train all models (MLflow tracking runs automatically)
make train

# 4a — Start API  →  http://localhost:8000
make api

# 4b — Start frontend  →  http://localhost:8501  (new terminal)
make app

# 4c — MLflow UI  →  http://localhost:5000  (optional, new terminal)
make mlflow
```

Alternatively, run the full stack via Docker:

```bash
docker compose up --build
# API:       http://localhost:8000
# Frontend:  http://localhost:8501
```

List all available commands:

```
make help
```

---

## Model Results

Trained on 600 synthetic laptop records (80/20 split, 5-fold CV):

| Model | R² | RMSE | MAE | MAPE | CV-R² |
|---|---|---|---|---|---|
| **LinearRegression** | **0.850** | **€123** | **€106** | **6.8 %** | 0.881 |
| LightGBM (Optuna) | 0.809 | €139 | €114 | 7.5 % | 0.832 |
| XGBoost (Optuna) | 0.772 | €152 | €123 | 7.9 % | 0.829 |
| RandomForest | 0.729 | €165 | €128 | 8.4 % | 0.828 |

> **Note:** LinearRegression leads because the synthetic dataset was intentionally generated with a linear relationship. With real scraped data, tree-based models (LightGBM, XGBoost) are expected to outperform it — MLflow tracking makes this comparison immediately reproducible after a data update.

The best model is automatically registered as `LaptopPriceModel@Production` in the MLflow Model Registry.

---

## Feature Engineering

`LaptopFeatureEngineer` is sklearn-compatible (`fit` / `transform`) and generates 19 features:

| Feature | Type | Description |
|---|---|---|
| `cpu_score` | Numeric | Regex-based CPU tier (i3 → 3.0 … M3 Pro → 9.0) |
| `gpu_score` | Numeric | GPU tier (integrated → 1.5 … RTX 4080 → 9.5) |
| `ram_gb` | Numeric | RAM in GB |
| `storage_gb` | Numeric | Storage capacity in GB |
| `display_inch` | Numeric | Display size in inches |
| `total_pixels` | Numeric | Width × height parsed from resolution string |
| `os_code` | Ordinal | Windows → 1, macOS → 2, Linux → 3, No OS → 0 |
| `is_ssd` | Binary | 1 if SSD, else 0 |
| `brand_*` | One-Hot | Top-10 brands (Apple, Lenovo, HP, Dell, …) |

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand":        "Lenovo",
    "cpu":          "Intel Core i7-13700H",
    "gpu":          "NVIDIA RTX 4060",
    "ram_gb":       16,
    "storage_gb":   512,
    "storage_type": "SSD",
    "display_inch": 15.6,
    "resolution":   "1920x1080",
    "os":           "Windows 11 Home",
    "weight_kg":    1.8
  }'
```

**Response:**

```json
{
  "predicted_price": 1644.32,
  "confidence_interval": [1479.89, 1808.75],
  "shap_values": {
    "cpu_score": 554.3,
    "gpu_score": 507.2,
    "ram_gb": 199.8
  },
  "base_value": 172.38,
  "shap_plot_base64": "iVBOR...",
  "feature_values": { "..." }
}
```

### All Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Price forecast + SHAP values + Waterfall plot |
| `/health` | GET | API health status |
| `/models` | GET | All MLflow runs with metrics |
| `/shap/summary` | GET | Global SHAP plots (Summary + Bar) |

Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Data Collection

The project supports two data sources:

**Synthetic data (default):**
```bash
python -m scraper.generate_dataset --n 600
```
Generates a realistic but artificially created dataset — sufficient for demonstrating the full pipeline.

**Web scraping (notebooksbilliger.de):**
```bash
# Connectivity test (run from a home network — datacenter IPs are blocked)
make scrape-check

# Run scraper
make scrape-nbb-fast   # fast: listing pages only
make scrape-nbb        # full: listing + detail pages
```

> **Note:** notebooksbilliger.de blocks datacenter and VPN IPs via bot protection. The scraper must be run from a residential IP (home broadband or mobile hotspot). Stealth mode via `playwright-stealth` is already integrated.

---

## Project Structure

```
projekt_01_preisvorhersage/
├── scraper/
│   ├── config.py                  # URLs, selectors, column names
│   ├── generate_dataset.py        # Synthetic dataset generator
│   ├── mediamarkt_scraper.py      # MediaMarkt scraper (experimental)
│   └── nbb_scraper.py             # notebooksbilliger.de Playwright scraper
├── data/
│   └── raw/
│       └── laptops_synthetic_2026-06-15.csv
├── src/
│   ├── features.py        # LaptopFeatureEngineer
│   ├── train.py           # MLflow experiment (4 models + Optuna)
│   ├── predict.py         # Inference module
│   └── shap_analysis.py   # SHAP explainability (Base64 plots)
├── api/
│   └── main.py            # FastAPI backend
├── app/
│   └── streamlit_app.py   # Streamlit frontend
├── tests/                 # pytest (45 tests)
├── .github/workflows/     # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Scraping | Playwright, playwright-stealth, BeautifulSoup4 |
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM, Optuna |
| Tracking | MLflow (Experiment Tracking + Model Registry) |
| Explainability | SHAP |
| API | FastAPI, Pydantic, uvicorn |
| Frontend | Streamlit |
| Testing | pytest |
| Infra | Docker, Docker Compose, GitHub Actions |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

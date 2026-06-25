# Laptop-Preisvorhersage — End-to-End ML Pipeline

[![CI](https://github.com/<username>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<username>/<repo>/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.19-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

End-to-End ML-Projekt: Laptop-Preise per Web-Scraping erfassen, vier Regressionsmodelle mit MLflow vergleichen, das beste Modell automatisch als Production-Alias registrieren, Vorhersagen per FastAPI bereitstellen und SHAP-Erklärungen in einem Streamlit-Dashboard visualisieren — vollständig containerisiert mit Docker.

---

## Architektur

```
┌────────────────────────────────┐
│        Datenbeschaffung        │
│  nbb_scraper.py  (Playwright)  │
│  generate_dataset.py           │
└───────────────┬────────────────┘
                │
                ▼
┌───────────────────────────────────────────────┐
│              Feature Engineering               │
│  LaptopFeatureEngineer  (sklearn-kompatibel)  │
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
                       │  bestes Modell → @Production
                       ▼
     ┌─────────────────────────────────────────┐
     │           FastAPI Backend               │
     │  POST /predict · GET /models · /health  │
     │  SHAP Waterfall-Plot (Base64)           │
     └─────────────────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Streamlit Frontend   │
              │   Preisprognose + CI   │
              │   SHAP Analyse         │
              └────────────────────────┘
```

---

## Voraussetzungen

- Python 3.12
- Docker & Docker Compose (für den Container-Stack)
- Git
- `make` (unter Windows: via [Git Bash](https://gitforwindows.org/) oder WSL2)

---

## Quickstart

```bash
# 1 — Repository klonen
git clone https://github.com/<username>/<repo>.git
cd <repo>

# 2 — Umgebung einrichten (venv + Playwright-Browser)
make setup

# 3 — Modelle trainieren (MLflow-Tracking läuft automatisch)
make train

# 4a — API starten  →  http://localhost:8000
make api

# 4b — Frontend starten  →  http://localhost:8501  (neues Terminal)
make app

# 4c — MLflow UI  →  http://localhost:5000  (optional, neues Terminal)
make mlflow
```

Alternativ als vollständiger Docker-Stack:

```bash
docker compose up --build
# API:       http://localhost:8000
# Frontend:  http://localhost:8501
```

Alle verfügbaren Kommandos:

```
make help
```

---

## Modell-Ergebnisse

Trainiert auf 600 synthetischen Laptop-Datensätzen (80/20 Split, 5-Fold CV):

| Modell | R² | RMSE | MAE | MAPE | CV-R² |
|---|---|---|---|---|---|
| **LinearRegression** | **0.850** | **123 €** | **106 €** | **6.8 %** | 0.881 |
| LightGBM (Optuna) | 0.809 | 139 € | 114 € | 7.5 % | 0.832 |
| XGBoost (Optuna) | 0.772 | 152 € | 123 € | 7.9 % | 0.829 |
| RandomForest | 0.729 | 165 € | 128 € | 8.4 % | 0.828 |

> **Hinweis:** LinearRegression führt hier, weil der synthetische Datensatz bewusst linear generiert wurde. Mit echten Scraping-Daten werden baumbasierte Modelle (LightGBM, XGBoost) erwartungsgemäß besser abschneiden — das Modell-Tracking via MLflow macht diesen Vergleich nach einem Datenaustausch sofort reproduzierbar.

Das beste Modell wird automatisch als `LaptopPriceModel@Production` in der MLflow Model Registry registriert.

---

## Feature Engineering

`LaptopFeatureEngineer` ist sklearn-kompatibel (`fit` / `transform`) und erzeugt 19 Features:

| Feature | Typ | Beschreibung |
|---|---|---|
| `cpu_score` | Numerisch | Regex-basiertes CPU-Tier (i3 → 3.0 … M3 Pro → 9.0) |
| `gpu_score` | Numerisch | GPU-Tier (integriert → 1.5 … RTX 4080 → 9.5) |
| `ram_gb` | Numerisch | Arbeitsspeicher in GB |
| `storage_gb` | Numerisch | Speichergröße in GB |
| `display_inch` | Numerisch | Displaygröße in Zoll |
| `total_pixels` | Numerisch | Breite × Höhe aus Auflösungsstring |
| `os_code` | Ordinal | Windows → 1, macOS → 2, Linux → 3, Ohne OS → 0 |
| `is_ssd` | Binär | 1 wenn SSD, sonst 0 |
| `brand_*` | One-Hot | Top-10-Marken (Apple, Lenovo, HP, Dell, …) |

---

## API-Referenz

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

### Alle Endpunkte

| Endpunkt | Methode | Beschreibung |
|---|---|---|
| `/predict` | POST | Preisprognose + SHAP-Werte + Waterfall-Plot |
| `/health` | GET | API-Status |
| `/models` | GET | Alle MLflow-Runs mit Metriken |
| `/shap/summary` | GET | Globale SHAP-Plots (Summary + Bar) |

Interaktive Dokumentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Datenbeschaffung

Das Projekt enthält zwei Wege zur Datenbeschaffung:

**Synthetische Daten (Standard):**
```bash
python -m scraper.generate_dataset --n 600
```
Erzeugt einen realistischen, aber künstlich generierten Datensatz — ausreichend für die Pipeline-Demonstration.

**Web-Scraping (notebooksbilliger.de):**
```bash
# Verbindungstest (von Heimnetz ausführen — Datacenter-IPs werden geblockt)
make scrape-check

# Scraper starten
make scrape-nbb-fast   # schnell: nur Listing-Seiten
make scrape-nbb        # vollständig: Listing + Detailseiten
```

> **Hinweis:** notebooksbilliger.de blockiert Datacenter- und VPN-IPs per Bot-Protection. Der Scraper muss von einer Residential-IP (Heimnetz oder Mobilfunk-Hotspot) ausgeführt werden. Der Stealth-Modus via `playwright-stealth` ist bereits integriert.

---

## Projektstruktur

```
projekt_01_preisvorhersage/
├── scraper/
│   ├── config.py                  # URLs, Selektoren, Spaltennamen
│   ├── generate_dataset.py        # Synthetischen Datensatz generieren
│   ├── mediamarkt_scraper.py      # MediaMarkt Scraper (experimentell)
│   └── nbb_scraper.py             # notebooksbilliger.de Playwright-Scraper
├── data/
│   └── raw/
│       └── laptops_synthetic_2026-06-15.csv
├── src/
│   ├── features.py        # LaptopFeatureEngineer
│   ├── train.py           # MLflow-Experiment (4 Modelle + Optuna)
│   ├── predict.py         # Inferenz-Modul
│   └── shap_analysis.py   # SHAP-Explainability (Base64-Plots)
├── api/
│   └── main.py            # FastAPI Backend
├── app/
│   └── streamlit_app.py   # Streamlit Frontend
├── tests/                 # pytest (45 Tests)
├── .github/workflows/     # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Tech Stack

| Schicht | Technologien |
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

## Lizenz

MIT License — siehe [LICENSE](LICENSE) für Details.

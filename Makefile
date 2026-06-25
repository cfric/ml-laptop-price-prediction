PYTHON    := .venv/bin/python
PIP       := .venv/bin/pip
UVICORN   := .venv/bin/uvicorn
STREAMLIT := .venv/bin/streamlit
COMPOSE   := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Laptop Price Prediction — available commands:"
	@echo ""
	@echo "  make setup        Create virtualenv and install dependencies"
	@echo "  make train        Train all models (MLflow tracking)"
	@echo "  make api          Start FastAPI backend  (localhost:8000)"
	@echo "  make app          Start Streamlit frontend  (localhost:8501)"
	@echo "  make mlflow       Start MLflow UI  (localhost:5000)"
	@echo "  make test         Run test suite"
	@echo "  make scrape-check    Test NBB connectivity (run from Heimnetzwerk!)"
	@echo "  make scrape-nbb      Run NBB scraper (10 pages, with detail pages)"
	@echo "  make scrape-nbb-fast Run NBB scraper (10 pages, listing-only, faster)"
	@echo "  make scrape-mm       Run MediaMarkt scraper"
	@echo "  make docker-up    Build and start full stack via Docker Compose"
	@echo "  make docker-down  Stop Docker Compose stack"
	@echo "  make clean        Remove generated artifacts"

# ─── Setup ────────────────────────────────────────────────────────────────────

.PHONY: setup
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m playwright install chromium

# ─── Core pipeline ────────────────────────────────────────────────────────────

.PHONY: train
train:
	$(PYTHON) -m src.train

.PHONY: api
api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: app
app:
	$(STREAMLIT) run app/streamlit_app.py --server.port 8501

.PHONY: mlflow
mlflow:
	$(PYTHON) -m mlflow ui --port 5000

# ─── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

# ─── Scraping ─────────────────────────────────────────────────────────────────

.PHONY: scrape-check
scrape-check:
	$(PYTHON) -m scraper.nbb_scraper --check

.PHONY: scrape-nbb
scrape-nbb:
	$(PYTHON) -m scraper.nbb_scraper --max-pages 10

.PHONY: scrape-nbb-fast
scrape-nbb-fast:
	$(PYTHON) -m scraper.nbb_scraper --max-pages 10 --listing-only

.PHONY: scrape-mm
scrape-mm:
	$(PYTHON) -m scraper.mediamarkt_scraper

.PHONY: scrape
scrape: scrape-nbb

# ─── Docker ───────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	$(COMPOSE) up --build

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

# ─── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf data/processed/ mlruns/ mlartifacts/ logs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

"""
MLflow Experiment: 4 Modelle trainieren und vergleichen.
Verwendung: python -m src.train [--data PATH] [--test-size 0.2] [--trials N]
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.features import LaptopFeatureEngineer, PROCESSED_DIR
from src.shap_analysis import SHAPAnalyzer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

EXPERIMENT_NAME = "laptop_price_prediction"
REGISTRY_NAME   = "LaptopPriceModel"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse":  rmse,
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "r2":    float(r2_score(y_true, y_pred)),
        "mape":  _mape(np.array(y_true), np.array(y_pred)),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def _feature_importance_plot(model, feature_names: list[str], run_name: str) -> str:
    path = str(PROCESSED_DIR / f"feat_importance_{run_name}.png")
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return ""

    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx][::-1], imp[idx][::-1], color="#3B82F6")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {run_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _pred_vs_actual_plot(y_true, y_pred, run_name: str) -> str:
    path = str(PROCESSED_DIR / f"pred_vs_actual_{run_name}.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#3B82F6", s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Price (€)")
    ax.set_ylabel("Predicted Price (€)")
    ax.set_title(f"Predicted vs. Actual — {run_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ─── Optuna Tuning ────────────────────────────────────────────────────────────

def _tune_xgboost(X_train, y_train, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("XGBoost best CV-R²=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


def _tune_lightgbm(X_train, y_train, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("LightGBM best CV-R²=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


# ─── Single model run ─────────────────────────────────────────────────────────

def train_model(
    model,
    run_name: str,
    X_train, X_test, y_train, y_test,
    feature_names: list[str],
    params: dict,
) -> tuple[dict, str]:

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        # CV score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"]  = float(cv_scores.std())

        # Log to MLflow
        mlflow.log_params({**params, "n_features": len(feature_names), "n_train": len(X_train)})
        mlflow.log_metrics(metrics)

        # Artefakte
        fi_path = _feature_importance_plot(model, feature_names, run_name)
        pa_path = _pred_vs_actual_plot(y_test, y_pred, run_name)
        if fi_path:
            mlflow.log_artifact(fi_path, "plots")
        if pa_path:
            mlflow.log_artifact(pa_path, "plots")

        # Modell speichern
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

        log.info("  %-22s  RMSE=%.1f  MAE=%.1f  R²=%.3f  MAPE=%.1f%%",
                 run_name, metrics["rmse"], metrics["mae"], metrics["r2"], metrics["mape"])
        return metrics, run_id, model


# ─── Quantile Regression ──────────────────────────────────────────────────────

def train_quantile_models(X_train, y_train) -> tuple:
    """Trainiert 10%- und 90%-Quantil-Modelle für Konfidenzintervalle."""
    common = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  subsample=0.8, random_state=42)

    log.info("Training Quantile-Modelle (α=0.10 / α=0.90) …")
    q_low  = GradientBoostingRegressor(loss="quantile", alpha=0.10, **common)
    q_high = GradientBoostingRegressor(loss="quantile", alpha=0.90, **common)
    q_low.fit(X_train, y_train)
    q_high.fit(X_train, y_train)

    joblib.dump(q_low,  PROCESSED_DIR / "quantile_low.pkl")
    joblib.dump(q_high, PROCESSED_DIR / "quantile_high.pkl")
    log.info("Quantile-Modelle gespeichert → %s", PROCESSED_DIR)
    return q_low, q_high


# ─── Main training loop ───────────────────────────────────────────────────────

def run_experiment(data_path: Path, test_size: float = 0.2, n_trials: int = 30) -> str:
    # Load data
    df = pd.read_csv(data_path)
    log.info("Loaded %d rows from %s", len(df), data_path)

    # Drop rows without price
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 100]   # sanity filter
    log.info("After cleaning: %d rows", len(df))

    # Feature engineering
    eng = LaptopFeatureEngineer()
    X = eng.fit_transform(df)
    y = df["price"].values

    eng.save_config()
    joblib.dump(eng, PROCESSED_DIR / "feature_engineer.pkl")
    log.info("Features: %s", eng.feature_names_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Quantile-Modelle für Konfidenzintervalle
    train_quantile_models(X_train, y_train)

    # Experiment setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow experiment: %s", EXPERIMENT_NAME)

    # Optuna Tuning für XGBoost und LightGBM
    log.info("Starte Optuna-Tuning (%d Trials je Modell) …", n_trials)
    xgb_params = _tune_xgboost(X_train, y_train, n_trials=n_trials)
    lgbm_params = _tune_lightgbm(X_train, y_train, n_trials=n_trials)

    models = [
        (
            LinearRegression(),
            "LinearRegression",
            {},
        ),
        (
            RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
            "RandomForest",
            {"n_estimators": 300, "max_depth": "None"},
        ),
        (
            XGBRegressor(**xgb_params),
            "XGBoost_Tuned",
            xgb_params,
        ),
        (
            LGBMRegressor(**lgbm_params),
            "LightGBM_Tuned",
            lgbm_params,
        ),
    ]

    results: list[tuple[str, dict, str, object]] = []
    for model, name, params in models:
        log.info("Training %s …", name)
        metrics, run_id, fitted_model = train_model(
            model, name, X_train, X_test, y_train, y_test,
            eng.feature_names_, params,
        )
        results.append((name, metrics, run_id, fitted_model))

    # Best model → MLflow Registry
    best = max(results, key=lambda r: r[1]["r2"])
    best_name, best_metrics, best_run_id, best_model_obj = best
    log.info("Best model: %s  (R²=%.3f)", best_name, best_metrics["r2"])

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(REGISTRY_NAME, "Production", mv.version)
    log.info("Registered %s v%s as '%s@Production'", REGISTRY_NAME, mv.version, REGISTRY_NAME)

    # Production-Modell als joblib speichern (Docker-kompatibel, kein MLflow-Pfad nötig)
    joblib.dump(best_model_obj, PROCESSED_DIR / "production_model.pkl")
    log.info("Production model saved → %s/production_model.pkl", PROCESSED_DIR)

    # SHAP Summary- und Bar-Plot über den Trainingsdatensatz
    log.info("Generiere SHAP Summary-Plots …")
    best_model = best_model_obj
    background = X_train[:50]  # repräsentatives Background-Sample
    analyzer = SHAPAnalyzer(best_model, eng.feature_names_, background)

    summary_b64 = analyzer.summary_plot_base64(X_train)
    bar_b64     = analyzer.bar_plot_base64(X_train)

    summary_path = str(PROCESSED_DIR / "shap_summary.png")
    bar_path     = str(PROCESSED_DIR / "shap_bar.png")

    import base64, io
    with open(summary_path, "wb") as f:
        f.write(base64.b64decode(summary_b64))
    with open(bar_path, "wb") as f:
        f.write(base64.b64decode(bar_b64))

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(summary_path, "shap")
        mlflow.log_artifact(bar_path,     "shap")
    log.info("SHAP-Plots gespeichert → %s", PROCESSED_DIR)

    return best_run_id


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train laptop price models with MLflow")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to scraped CSV (default: newest file in data/raw/)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=30,
                        help="Anzahl Optuna-Trials je Modell (default: 30)")
    args = parser.parse_args()

    if args.data:
        data_path = Path(args.data)
    else:
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        csvs = [p for p in sorted(raw_dir.glob("*.csv")) if p.stat().st_size > 1024]
        if not csvs:
            raise FileNotFoundError(f"No non-empty CSV found in {raw_dir}. Run the scraper first.")
        data_path = csvs[-1]
        log.info("Auto-selected: %s", data_path)

    run_experiment(data_path, test_size=args.test_size, n_trials=args.trials)


if __name__ == "__main__":
    main()

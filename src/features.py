"""
Feature Engineering Pipeline für Laptop-Preisdaten.
Sklearn-kompatibel: fit() / transform() / fit_transform()
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── CPU scoring ─────────────────────────────────────────────────────────────

CPU_PATTERNS = [
    # Intel
    (r"core\s*i9|i9[-\s]", 9.0),
    (r"core\s*i7|i7[-\s]", 7.0),
    (r"core\s*i5|i5[-\s]", 5.0),
    (r"core\s*i3|i3[-\s]", 3.0),
    (r"core\s*ultra\s*9|ultra\s*9", 9.5),
    (r"core\s*ultra\s*7|ultra\s*7", 8.0),
    (r"core\s*ultra\s*5|ultra\s*5", 6.0),
    (r"pentium|celeron|n\d{4}", 1.5),
    # AMD
    (r"ryzen\s*9|r9[-\s]", 9.0),
    (r"ryzen\s*7|r7[-\s]", 7.0),
    (r"ryzen\s*5|r5[-\s]", 5.0),
    (r"ryzen\s*3|r3[-\s]", 3.0),
    (r"athlon", 1.5),
    # Apple
    (r"m[34]\s*(pro|max|ultra)", 9.5),
    (r"m[34]\b", 8.5),
    (r"m[12]\s*(pro|max|ultra)", 8.0),
    (r"m[12]\b", 7.0),
    # Qualcomm
    (r"snapdragon\s*x\s*(elite|plus)", 8.0),
    (r"snapdragon\s*x\b", 6.5),
]

# ─── GPU scoring ──────────────────────────────────────────────────────────────

GPU_PATTERNS = [
    # NVIDIA high
    (r"rtx\s*4090", 10.0),
    (r"rtx\s*4080", 9.5),
    (r"rtx\s*4070", 8.5),
    (r"rtx\s*4060", 7.5),
    (r"rtx\s*4050", 6.5),
    (r"rtx\s*3080", 8.0),
    (r"rtx\s*3070", 7.0),
    (r"rtx\s*3060", 6.0),
    (r"rtx\s*3050", 5.0),
    (r"gtx\s*1650|gtx\s*1660", 3.5),
    (r"gtx\s*1[0-9]{3}", 3.0),
    # AMD
    (r"rx\s*7[0-9]{3}m?", 7.0),
    (r"rx\s*6[0-9]{3}m?", 5.5),
    (r"rx\s*5[0-9]{3}m?", 3.5),
    # Intel Arc
    (r"arc\s*a[5-7]", 5.0),
    (r"arc\s*a[1-4]", 3.5),
    # Integrated
    (r"iris\s*xe|uhd|radeon\s*\d{3}[^m]|mali|apple\s*m", 1.5),
]


def _score_cpu(raw: str) -> float:
    if not isinstance(raw, str):
        return 0.0
    text = raw.lower()
    for pattern, score in CPU_PATTERNS:
        if re.search(pattern, text):
            return score
    return 2.0  # generic / unknown


def _score_gpu(raw: str) -> float:
    if not isinstance(raw, str):
        return 0.0
    text = raw.lower()
    for pattern, score in GPU_PATTERNS:
        if re.search(pattern, text):
            return score
    return 1.0  # assume integrated if nothing matches


def _parse_ram(val) -> float:
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"(\d+)", val)
        if m:
            return float(m.group(1))
    return np.nan


def _parse_storage(val) -> float:
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(GB|TB)", val, re.I)
        if m:
            v = float(m.group(1).replace(",", "."))
            return v * 1024 if m.group(2).upper() == "TB" else v
    return np.nan


def _parse_display(val) -> float:
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"(\d+[.,]\d+)", val)
        if m:
            return float(m.group(1).replace(",", "."))
    return np.nan


def _resolution_pixels(val) -> float:
    """'1920x1080' / '2560 x 1600' → 1920*1080 = 2073600"""
    if not isinstance(val, str):
        return np.nan
    m = re.search(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", val, re.I)
    if m:
        return float(m.group(1)) * float(m.group(2))
    return np.nan


def _encode_os(val: str) -> int:
    if not isinstance(val, str):
        return 0
    v = val.lower()
    if "windows" in v:
        return 1
    if "mac" in v or "macos" in v:
        return 2
    if "linux" in v or "chrome" in v or "ubuntu" in v:
        return 3
    return 0  # Other / no OS


KNOWN_BRANDS = [
    "apple", "lenovo", "hp", "dell", "asus", "acer",
    "microsoft", "samsung", "msi", "lg",
]


class LaptopFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transform raw scraped laptop DataFrame into a numeric feature matrix.

    Output features (in order):
        cpu_score, gpu_score, ram_gb, storage_gb, display_inch,
        total_pixels, os_code, is_ssd,
        brand_<name> × N  (one-hot, top brands + "other")
    """

    def __init__(self):
        self.brand_columns_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        brands_in_data = X["brand"].dropna().str.lower().str.strip()
        top_brands = [b for b in KNOWN_BRANDS if (brands_in_data == b).any()]
        self.brand_columns_ = [f"brand_{b}" for b in top_brands] + ["brand_other"]

        self.feature_names_ = [
            "cpu_score", "gpu_score", "ram_gb", "storage_gb",
            "display_inch", "total_pixels", "os_code", "is_ssd",
        ] + self.brand_columns_

        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        df = X.copy()

        cpu_score   = df["cpu"].apply(_score_cpu)
        gpu_score   = df["gpu"].apply(_score_gpu)
        ram         = df["ram_gb"].apply(_parse_ram)
        storage     = df["storage_gb"].apply(_parse_storage)
        display     = df["display_inch"].apply(_parse_display)
        pixels      = df.get("resolution", pd.Series(np.nan, index=df.index)).apply(_resolution_pixels)
        os_code     = df.get("os", pd.Series("", index=df.index)).apply(_encode_os)
        is_ssd      = df.get("storage_type", pd.Series("", index=df.index)).apply(
            lambda v: 1 if isinstance(v, str) and "ssd" in v.lower() else 0
        )

        base = np.column_stack([cpu_score, gpu_score, ram, storage, display, pixels, os_code, is_ssd])

        # Brand one-hot
        brand_lower = df["brand"].fillna("").str.lower().str.strip()
        brand_matrix = np.zeros((len(df), len(self.brand_columns_)), dtype=float)
        for i, col in enumerate(self.brand_columns_[:-1]):          # skip "brand_other"
            brand_name = col.replace("brand_", "")
            brand_matrix[:, i] = (brand_lower == brand_name).astype(float)
        # "other" = not any known brand
        known_mask = brand_matrix[:, :-1].sum(axis=1) > 0
        brand_matrix[:, -1] = (~known_mask.astype(bool)).astype(float)

        return np.hstack([base, brand_matrix])

    def save_config(self, path: Path | None = None):
        path = path or (PROCESSED_DIR / "feature_config.json")
        config = {
            "feature_names": self.feature_names_,
            "brand_columns": self.brand_columns_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: Path | None = None) -> "LaptopFeatureEngineer":
        path = path or (PROCESSED_DIR / "feature_config.json")
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        eng = cls()
        eng.feature_names_ = config["feature_names"]
        eng.brand_columns_ = config["brand_columns"]
        return eng

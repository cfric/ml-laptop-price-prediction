import pandas as pd
import pytest


@pytest.fixture
def sample_specs():
    return {
        "brand": "Dell",
        "cpu": "Intel Core i7-13700H",
        "gpu": "NVIDIA RTX 4060",
        "ram_gb": 16,
        "storage_gb": 512,
        "storage_type": "SSD",
        "display_inch": 15.6,
        "resolution": "1920x1080",
        "os": "Windows 11 Home",
        "weight_kg": 1.8,
    }


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame([
        {"brand": "Dell",    "cpu": "Intel Core i7-13700H", "gpu": "NVIDIA RTX 4060",
         "ram_gb": 16,  "storage_gb": 512,  "storage_type": "SSD", "display_inch": 15.6,
         "resolution": "1920x1080", "os": "Windows 11 Home", "weight_kg": 1.8},
        {"brand": "Apple",   "cpu": "Apple M3 Pro",          "gpu": "Apple GPU",
         "ram_gb": 18,  "storage_gb": 512,  "storage_type": "SSD", "display_inch": 14.2,
         "resolution": "3024x1964", "os": "macOS",            "weight_kg": 1.6},
        {"brand": "Lenovo",  "cpu": "AMD Ryzen 5 7530U",    "gpu": "AMD Radeon 610M",
         "ram_gb": 8,   "storage_gb": 256,  "storage_type": "SSD", "display_inch": 14.0,
         "resolution": "1920x1080", "os": "Windows 11",      "weight_kg": 1.5},
        {"brand": "HP",      "cpu": "Intel Core i5-1235U",  "gpu": "Intel Iris Xe",
         "ram_gb": 8,   "storage_gb": 256,  "storage_type": "SSD", "display_inch": 13.3,
         "resolution": "2560x1600", "os": "Windows 11 Home", "weight_kg": 1.3},
        {"brand": "MSI",     "cpu": "Intel Core i9-13980HX","gpu": "NVIDIA RTX 4090",
         "ram_gb": 64,  "storage_gb": 2048, "storage_type": "SSD", "display_inch": 17.3,
         "resolution": "2560x1440", "os": "Windows 11 Pro",  "weight_kg": 3.1},
    ])

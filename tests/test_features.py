import numpy as np
import pandas as pd
import pytest

from src.features import (
    LaptopFeatureEngineer,
    _encode_os,
    _parse_display,
    _parse_ram,
    _parse_storage,
    _resolution_pixels,
    _score_cpu,
    _score_gpu,
)


class TestScoreCpu:
    def test_intel_hierarchy(self):
        assert _score_cpu("Intel Core i3") < _score_cpu("Intel Core i5")
        assert _score_cpu("Intel Core i5") < _score_cpu("Intel Core i7")
        assert _score_cpu("Intel Core i7") < _score_cpu("Intel Core i9")

    def test_intel_ultra(self):
        assert _score_cpu("Intel Core Ultra 7") > _score_cpu("Intel Core i7")

    def test_amd_ryzen(self):
        assert _score_cpu("AMD Ryzen 5") < _score_cpu("AMD Ryzen 7")
        assert _score_cpu("AMD Ryzen 7") < _score_cpu("AMD Ryzen 9")

    def test_apple_m3_pro_highest(self):
        assert _score_cpu("Apple M3 Pro") >= _score_cpu("Intel Core i9")

    def test_unknown_returns_fallback(self):
        assert _score_cpu("Unknown CPU XYZ") == 2.0

    def test_non_string_returns_zero(self):
        assert _score_cpu(None) == 0.0
        assert _score_cpu(42) == 0.0


class TestScoreGpu:
    def test_nvidia_hierarchy(self):
        assert _score_gpu("NVIDIA RTX 4050") < _score_gpu("NVIDIA RTX 4060")
        assert _score_gpu("NVIDIA RTX 4060") < _score_gpu("NVIDIA RTX 4070")
        assert _score_gpu("NVIDIA RTX 4070") < _score_gpu("NVIDIA RTX 4080")
        assert _score_gpu("NVIDIA RTX 4080") < _score_gpu("NVIDIA RTX 4090")

    def test_integrated_lower_than_dedicated(self):
        assert _score_gpu("Intel Iris Xe") < _score_gpu("NVIDIA RTX 4050")

    def test_unknown_returns_fallback(self):
        assert _score_gpu("Unknown GPU") == 1.0

    def test_non_string_returns_zero(self):
        assert _score_gpu(None) == 0.0


class TestParseRam:
    def test_numeric_passthrough(self):
        assert _parse_ram(16) == 16.0
        assert _parse_ram(8.0) == 8.0

    def test_string_with_unit(self):
        assert _parse_ram("32 GB") == 32.0
        assert _parse_ram("16GB") == 16.0

    def test_invalid_returns_nan(self):
        assert np.isnan(_parse_ram("keine Angabe"))
        assert np.isnan(_parse_ram(None))


class TestParseStorage:
    def test_numeric_passthrough(self):
        assert _parse_storage(512) == 512.0

    def test_gb_string(self):
        assert _parse_storage("256 GB") == 256.0
        assert _parse_storage("1024GB") == 1024.0

    def test_tb_conversion(self):
        assert _parse_storage("1 TB") == 1024.0
        assert _parse_storage("2TB") == 2048.0

    def test_invalid_returns_nan(self):
        assert np.isnan(_parse_storage("unbekannt"))


class TestParseDisplay:
    def test_numeric_passthrough(self):
        assert _parse_display(15.6) == 15.6

    def test_string_with_decimal(self):
        assert _parse_display("14.2 inch") == 14.2
        assert _parse_display("17,3 Zoll") == 17.3

    def test_invalid_returns_nan(self):
        assert np.isnan(_parse_display("groß"))


class TestResolutionPixels:
    def test_full_hd(self):
        assert _resolution_pixels("1920x1080") == 1920 * 1080

    def test_4k(self):
        assert _resolution_pixels("3840x2160") == 3840 * 2160

    def test_with_spaces(self):
        assert _resolution_pixels("2560 x 1600") == 2560 * 1600

    def test_invalid_returns_nan(self):
        assert np.isnan(_resolution_pixels("Full HD"))
        assert np.isnan(_resolution_pixels(None))


class TestEncodeOs:
    def test_windows_variants(self):
        assert _encode_os("Windows 11 Home") == 1
        assert _encode_os("Windows 11 Pro") == 1
        assert _encode_os("Windows") == 1

    def test_macos(self):
        assert _encode_os("macOS") == 2
        assert _encode_os("Mac OS X") == 2

    def test_linux(self):
        assert _encode_os("Linux") == 3
        assert _encode_os("Ubuntu 22.04") == 3

    def test_unknown_returns_zero(self):
        assert _encode_os("Ohne OS") == 0
        assert _encode_os("") == 0

    def test_non_string_returns_zero(self):
        assert _encode_os(None) == 0


class TestLaptopFeatureEngineer:
    def test_fit_transform_shape(self, sample_dataframe):
        eng = LaptopFeatureEngineer()
        X = eng.fit_transform(sample_dataframe)
        assert X.shape[0] == len(sample_dataframe)
        assert X.shape[1] == len(eng.feature_names_)

    def test_feature_names_contain_base_features(self, sample_dataframe):
        eng = LaptopFeatureEngineer()
        eng.fit(sample_dataframe)
        for name in ["cpu_score", "gpu_score", "ram_gb", "storage_gb",
                     "display_inch", "total_pixels", "os_code", "is_ssd"]:
            assert name in eng.feature_names_

    def test_brand_one_hot(self, sample_dataframe):
        eng = LaptopFeatureEngineer()
        X = eng.fit_transform(sample_dataframe)
        brand_cols = [i for i, n in enumerate(eng.feature_names_) if n.startswith("brand_")]
        # Jede Zeile hat genau eine 1 in den Brand-Spalten
        brand_sums = X[:, brand_cols].sum(axis=1)
        assert np.all(brand_sums == 1.0)

    def test_transform_single_row(self, sample_dataframe):
        eng = LaptopFeatureEngineer()
        eng.fit(sample_dataframe)
        single = sample_dataframe.iloc[[0]]
        X = eng.transform(single)
        assert X.shape == (1, len(eng.feature_names_))

    def test_ssd_flag(self, sample_dataframe):
        eng = LaptopFeatureEngineer()
        X = eng.fit_transform(sample_dataframe)
        is_ssd_idx = eng.feature_names_.index("is_ssd")
        # Alle Zeilen haben storage_type=SSD → is_ssd sollte 1 sein
        assert np.all(X[:, is_ssd_idx] == 1.0)

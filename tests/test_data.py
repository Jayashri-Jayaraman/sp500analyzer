"""Tests for the data layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    SECTOR_MAP,
    SECTORS,
    TICKERS,
    compute_returns,
    generate_synthetic_prices,
    get_sector_series,
    load_prices,
    save_prices,
)


# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(tickers=["AAPL", "MSFT", "XOM"], years=2, seed=99)


@pytest.fixture
def small_prices() -> pd.DataFrame:
    """Minimal hand-crafted prices for deterministic tests."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {"A": [100.0, 102.0, 101.0, 105.0, 103.0], "B": [50.0, 51.0, 49.0, 52.0, 53.0]},
        index=dates,
    )


# ──────────────────────────────────────────────
#  Sector map
# ──────────────────────────────────────────────
class TestSectorMap:
    def test_all_tickers_have_sectors(self) -> None:
        for t in TICKERS:
            assert t in SECTOR_MAP

    def test_sectors_list_matches_map(self) -> None:
        assert set(SECTORS) == set(SECTOR_MAP.values())

    def test_get_sector_series_type(self) -> None:
        s = get_sector_series()
        assert isinstance(s, pd.Series)
        assert len(s) == len(TICKERS)


# ──────────────────────────────────────────────
#  Synthetic data generation
# ──────────────────────────────────────────────
class TestGenerateSyntheticPrices:
    def test_shape(self, prices: pd.DataFrame) -> None:
        assert prices.shape[1] == 3
        assert prices.shape[0] == 252 * 2

    def test_columns(self, prices: pd.DataFrame) -> None:
        assert list(prices.columns) == ["AAPL", "MSFT", "XOM"]

    def test_index_is_datetime(self, prices: pd.DataFrame) -> None:
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_no_nans(self, prices: pd.DataFrame) -> None:
        assert prices.isna().sum().sum() == 0

    def test_all_positive(self, prices: pd.DataFrame) -> None:
        assert (prices > 0).all().all()

    def test_deterministic_with_seed(self) -> None:
        a = generate_synthetic_prices(tickers=["AAPL"], years=1, seed=42)
        b = generate_synthetic_prices(tickers=["AAPL"], years=1, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = generate_synthetic_prices(tickers=["AAPL"], years=1, seed=1)
        b = generate_synthetic_prices(tickers=["AAPL"], years=1, seed=2)
        assert not a.equals(b)

    def test_default_tickers(self) -> None:
        df = generate_synthetic_prices(years=1)
        assert len(df.columns) == len(TICKERS)


# ──────────────────────────────────────────────
#  I/O
# ──────────────────────────────────────────────
class TestIO:
    def test_save_and_load_roundtrip(self, prices: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        save_prices(prices, path)
        loaded = load_prices(path)
        pd.testing.assert_frame_equal(prices, loaded, atol=0.01, check_freq=False)
        path.unlink()


# ──────────────────────────────────────────────
#  Returns
# ──────────────────────────────────────────────
class TestComputeReturns:
    def test_log_returns_shape(self, small_prices: pd.DataFrame) -> None:
        r = compute_returns(small_prices, method="log")
        assert r.shape == (4, 2)  # one row lost

    def test_simple_returns_shape(self, small_prices: pd.DataFrame) -> None:
        r = compute_returns(small_prices, method="simple")
        assert r.shape == (4, 2)

    def test_log_return_value(self, small_prices: pd.DataFrame) -> None:
        r = compute_returns(small_prices, method="log")
        expected = np.log(102.0 / 100.0)
        assert abs(r["A"].iloc[0] - expected) < 1e-10

    def test_simple_return_value(self, small_prices: pd.DataFrame) -> None:
        r = compute_returns(small_prices, method="simple")
        expected = (102.0 - 100.0) / 100.0
        assert abs(r["A"].iloc[0] - expected) < 1e-10

    def test_no_nans_in_output(self, small_prices: pd.DataFrame) -> None:
        r = compute_returns(small_prices, method="log")
        assert r.isna().sum().sum() == 0

    def test_invalid_method_raises(self, small_prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compute_returns(small_prices, method="invalid")

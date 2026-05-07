"""Tests for sector analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import SECTORS, compute_returns, generate_synthetic_prices
from src.sectors import (
    sector_correlation_matrix,
    sector_cumulative_returns,
    sector_momentum,
    sector_performance_summary,
    sector_relative_strength,
    sector_returns,
    sector_rolling_volatility,
    sector_ticker_counts,
    sector_weight_by_price,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(years=2, seed=42)


@pytest.fixture
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return compute_returns(prices, method="log")


class TestSectorReturns:
    def test_output_columns_are_sectors(self, returns: pd.DataFrame) -> None:
        sr = sector_returns(returns)
        assert set(sr.columns) == set(SECTORS)

    def test_output_length(self, returns: pd.DataFrame) -> None:
        sr = sector_returns(returns)
        assert len(sr) == len(returns)

    def test_no_nans(self, returns: pd.DataFrame) -> None:
        sr = sector_returns(returns)
        assert sr.isna().sum().sum() == 0


class TestSectorCumulativeReturns:
    def test_starts_near_one(self, returns: pd.DataFrame) -> None:
        sc = sector_cumulative_returns(returns)
        np.testing.assert_allclose(sc.iloc[0].values, 1.0, atol=0.05)

    def test_all_positive(self, returns: pd.DataFrame) -> None:
        sc = sector_cumulative_returns(returns)
        assert (sc > 0).all().all()


class TestSectorCorrelation:
    def test_shape(self, returns: pd.DataFrame) -> None:
        corr = sector_correlation_matrix(returns)
        n = len(SECTORS)
        assert corr.shape == (n, n)

    def test_diagonal_is_one(self, returns: pd.DataFrame) -> None:
        corr = sector_correlation_matrix(returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_bounded(self, returns: pd.DataFrame) -> None:
        corr = sector_correlation_matrix(returns)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()


class TestSectorRollingVol:
    def test_output_columns(self, returns: pd.DataFrame) -> None:
        vol = sector_rolling_volatility(returns, window=63)
        assert set(vol.columns) == set(SECTORS)

    def test_all_positive_after_window(self, returns: pd.DataFrame) -> None:
        vol = sector_rolling_volatility(returns, window=63).dropna()
        assert (vol >= 0).all().all()


class TestSectorPerformance:
    def test_output_columns(self, returns: pd.DataFrame, prices: pd.DataFrame) -> None:
        perf = sector_performance_summary(returns, prices)
        assert "annual_return" in perf.columns
        assert "sharpe" in perf.columns
        assert "max_drawdown" in perf.columns

    def test_output_index_is_sectors(self, returns: pd.DataFrame, prices: pd.DataFrame) -> None:
        perf = sector_performance_summary(returns, prices)
        assert set(perf.index) == set(SECTORS)


class TestSectorMomentum:
    def test_output_columns(self, returns: pd.DataFrame) -> None:
        mom = sector_momentum(returns, lookback=63)
        assert set(mom.columns) == set(SECTORS)

    def test_nan_before_lookback(self, returns: pd.DataFrame) -> None:
        mom = sector_momentum(returns, lookback=63)
        assert mom.iloc[:62].isna().all().all()


class TestSectorRelativeStrength:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        rs = sector_relative_strength(returns, lookback=63)
        assert rs.shape[1] == len(SECTORS)


class TestSectorTickerCounts:
    def test_sum_matches_total(self) -> None:
        counts = sector_ticker_counts()
        from src.data import TICKERS

        assert counts.sum() == len(TICKERS)

    def test_all_sectors_present(self) -> None:
        counts = sector_ticker_counts()
        assert set(counts.index) == set(SECTORS)


class TestSectorWeights:
    def test_rows_sum_to_one(self, prices: pd.DataFrame) -> None:
        w = sector_weight_by_price(prices)
        np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-10)

    def test_all_positive(self, prices: pd.DataFrame) -> None:
        w = sector_weight_by_price(prices)
        assert (w >= 0).all().all()

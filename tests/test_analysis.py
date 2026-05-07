"""Tests for the vectorised analysis engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    beta,
    correlation_matrix,
    cvar,
    drawdown,
    ewm_volatility,
    max_drawdown,
    rolling_correlation,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    summary_stats,
    value_at_risk,
)
from src.data import compute_returns, generate_synthetic_prices


@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(tickers=["AAPL", "MSFT", "XOM", "JPM"], years=3, seed=42)


@pytest.fixture
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return compute_returns(prices, method="log")


@pytest.fixture
def deterministic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame(
        {
            "UP": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "DOWN": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            "FLAT": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "VEE": [100, 80, 60, 40, 50, 60, 80, 100, 120, 140],
        },
        index=dates,
        dtype=float,
    )


class TestRollingVolatility:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        vol = rolling_volatility(returns, window=21)
        assert vol.shape == returns.shape

    def test_first_window_is_nan(self, returns: pd.DataFrame) -> None:
        vol = rolling_volatility(returns, window=21)
        assert vol.iloc[:20].isna().all().all()

    def test_after_window_not_nan(self, returns: pd.DataFrame) -> None:
        vol = rolling_volatility(returns, window=21)
        assert vol.iloc[21:].notna().all().all()

    def test_annualisation(self, returns: pd.DataFrame) -> None:
        vol_ann = rolling_volatility(returns, window=21, annualise=True)
        vol_raw = rolling_volatility(returns, window=21, annualise=False)
        ratio = (vol_ann / vol_raw).dropna()
        np.testing.assert_allclose(ratio.values, np.sqrt(252), atol=1e-10)

    def test_all_positive(self, returns: pd.DataFrame) -> None:
        vol = rolling_volatility(returns, window=21).dropna()
        assert (vol >= 0).all().all()


class TestEwmVolatility:
    def test_output_shape(self, returns: pd.DataFrame) -> None:
        vol = ewm_volatility(returns, span=21)
        assert vol.shape == returns.shape

    def test_all_positive(self, returns: pd.DataFrame) -> None:
        vol = ewm_volatility(returns, span=21).dropna()
        assert (vol >= 0).all().all()


class TestCorrelation:
    def test_matrix_shape(self, returns: pd.DataFrame) -> None:
        corr = correlation_matrix(returns)
        n = len(returns.columns)
        assert corr.shape == (n, n)

    def test_diagonal_is_one(self, returns: pd.DataFrame) -> None:
        corr = correlation_matrix(returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self, returns: pd.DataFrame) -> None:
        corr = correlation_matrix(returns)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_bounded(self, returns: pd.DataFrame) -> None:
        corr = correlation_matrix(returns)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_rolling_correlation_output(self, returns: pd.DataFrame) -> None:
        rc = rolling_correlation(returns, "AAPL", "MSFT", window=63)
        assert isinstance(rc, pd.Series)
        assert len(rc) == len(returns)


class TestDrawdown:
    def test_always_non_positive(self, prices: pd.DataFrame) -> None:
        dd = drawdown(prices)
        assert (dd <= 1e-10).all().all()

    def test_starts_at_zero(self, deterministic_prices: pd.DataFrame) -> None:
        dd = drawdown(deterministic_prices)
        assert (dd.iloc[0] == 0).all()

    def test_monotonic_up_zero_drawdown(self, deterministic_prices: pd.DataFrame) -> None:
        dd = drawdown(deterministic_prices)
        assert (dd["UP"] == 0).all()

    def test_monotonic_down_drawdown(self, deterministic_prices: pd.DataFrame) -> None:
        dd = drawdown(deterministic_prices)
        assert abs(dd["DOWN"].iloc[-1] - (-0.9)) < 1e-10

    def test_flat_zero_drawdown(self, deterministic_prices: pd.DataFrame) -> None:
        dd = drawdown(deterministic_prices)
        assert (dd["FLAT"] == 0).all()

    def test_max_drawdown_values(self, deterministic_prices: pd.DataFrame) -> None:
        mdd = max_drawdown(deterministic_prices)
        assert mdd["UP"] == 0
        assert abs(mdd["DOWN"] - (-0.9)) < 1e-10
        assert mdd["FLAT"] == 0
        assert abs(mdd["VEE"] - (-0.6)) < 1e-10


class TestRiskMetrics:
    def test_sharpe_type(self, returns: pd.DataFrame) -> None:
        sr = sharpe_ratio(returns)
        assert isinstance(sr, pd.Series)
        assert len(sr) == len(returns.columns)

    def test_sortino_type(self, returns: pd.DataFrame) -> None:
        so = sortino_ratio(returns)
        assert isinstance(so, pd.Series)

    def test_beta_of_market_is_one(self, returns: pd.DataFrame) -> None:
        market = returns.mean(axis=1)
        r2 = returns.copy()
        r2["MKT"] = market
        b = beta(r2, market)
        assert abs(b["MKT"] - 1.0) < 1e-10

    def test_var_is_negative(self, returns: pd.DataFrame) -> None:
        var = value_at_risk(returns, confidence=0.05)
        assert (var < 0).all()

    def test_cvar_le_var(self, returns: pd.DataFrame) -> None:
        var = value_at_risk(returns, confidence=0.05)
        cv = cvar(returns, confidence=0.05)
        assert (cv <= var + 1e-10).all()


class TestSummaryStats:
    def test_output_columns(self, returns: pd.DataFrame, prices: pd.DataFrame) -> None:
        stats = summary_stats(returns, prices)
        expected = [
            "annual_return",
            "annual_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "var_5pct",
            "cvar_5pct",
            "beta",
        ]
        assert list(stats.columns) == expected

    def test_output_rows(self, returns: pd.DataFrame, prices: pd.DataFrame) -> None:
        stats = summary_stats(returns, prices)
        assert list(stats.index) == list(returns.columns)

    def test_no_nans(self, returns: pd.DataFrame, prices: pd.DataFrame) -> None:
        stats = summary_stats(returns, prices)
        assert stats.isna().sum().sum() == 0

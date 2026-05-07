"""
Sector-level analysis — aggregations, performance, and rotation metrics.

All operations use Pandas groupby and vectorised arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import get_sector_series


# ──────────────────────────────────────────────
#  Sector returns
# ──────────────────────────────────────────────
def sector_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight daily sector returns.

    Groups ticker columns by sector, computes the daily mean return
    per sector.  Fully vectorised via groupby on axis=1.
    """
    sector_map: pd.Series = get_sector_series()
    # Align: only include tickers present in both
    common: list[str] = [t for t in returns.columns if t in sector_map.index]
    aligned: pd.DataFrame = returns[common]
    mapping: pd.Series = sector_map[common]

    # groupby on columns (axis=1)
    return aligned.T.groupby(mapping).mean().T


def sector_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative sector returns (growth of $1)."""
    sr: pd.DataFrame = sector_returns(returns)
    return (1 + sr).cumprod()


# ──────────────────────────────────────────────
#  Sector volatility
# ──────────────────────────────────────────────
def sector_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Annualised rolling volatility per sector."""
    sr: pd.DataFrame = sector_returns(returns)
    return sr.rolling(window=window, min_periods=window).std() * np.sqrt(252)


# ──────────────────────────────────────────────
#  Sector correlations
# ──────────────────────────────────────────────
def sector_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation matrix between sectors."""
    sr: pd.DataFrame = sector_returns(returns)
    return sr.corr()


# ──────────────────────────────────────────────
#  Sector performance table
# ──────────────────────────────────────────────
def sector_performance_summary(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summary statistics per sector.

    Computes annualised return, vol, Sharpe, max drawdown for each sector.
    """
    sr: pd.DataFrame = sector_returns(returns)
    sc: pd.DataFrame = sector_cumulative_returns(returns)

    # Max drawdown from cumulative return series
    peak: pd.DataFrame = sc.cummax()
    dd: pd.DataFrame = (sc - peak) / peak

    annual_return: pd.Series = sr.mean() * 252
    annual_vol: pd.Series = sr.std() * np.sqrt(252)
    sharpe: pd.Series = annual_return / annual_vol
    max_dd: pd.Series = dd.min()

    summary: pd.DataFrame = pd.DataFrame(
        {
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
    )
    return summary.round(4)


# ──────────────────────────────────────────────
#  Sector rotation / momentum
# ──────────────────────────────────────────────
def sector_momentum(
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Rolling sector momentum — trailing cumulative return over lookback window.

    Useful for sector rotation strategies.
    """
    sr: pd.DataFrame = sector_returns(returns)
    return sr.rolling(window=lookback, min_periods=lookback).sum()


def sector_relative_strength(
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Sector return relative to equal-weight market.

    Positive → sector outperforming market.
    Negative → sector underperforming.
    """
    sr: pd.DataFrame = sector_returns(returns)
    market: pd.Series = sr.mean(axis=1)
    mom: pd.DataFrame = sr.rolling(window=lookback, min_periods=lookback).sum()
    market_mom: pd.Series = market.rolling(window=lookback, min_periods=lookback).sum()
    return mom.sub(market_mom, axis=0)


# ──────────────────────────────────────────────
#  Sector composition
# ──────────────────────────────────────────────
def sector_ticker_counts() -> pd.Series:
    """Number of tickers per sector in the universe."""
    return get_sector_series().groupby(get_sector_series()).count().rename("count")


def sector_weight_by_price(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price-weighted sector allocation over time.

    Not market-cap weighted (we don't have shares outstanding),
    but shows how sector concentration evolves.
    """
    sector_map: pd.Series = get_sector_series()
    common: list[str] = [t for t in prices.columns if t in sector_map.index]
    aligned: pd.DataFrame = prices[common]
    mapping: pd.Series = sector_map[common]

    sector_sums: pd.DataFrame = aligned.T.groupby(mapping).sum().T
    total: pd.Series = sector_sums.sum(axis=1)
    return sector_sums.div(total, axis=0)

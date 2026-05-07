"""
Vectorised analysis engine — rolling volatility, correlations, risk metrics.

Every function operates on full DataFrames/Series using Pandas/NumPy
vectorised operations.  Zero Python-level loops over rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


#  Rolling volatility

def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling realised volatility.

    Parameters
    ----------
    returns   : daily log or simple returns (tickers as columns).
    window    : lookback in trading days (21 ≈ 1 month).
    annualise : multiply by √252 to get annualised vol.

    Returns
    -------
    DataFrame of same shape with rolling vol values.
    """
    vol: pd.DataFrame = returns.rolling(window=window, min_periods=window).std()
    if annualise:
        vol = vol * np.sqrt(252)
    return vol


def ewm_volatility(
    returns: pd.DataFrame,
    span: int = 21,
    annualise: bool = True,
) -> pd.DataFrame:
    """Exponentially weighted rolling volatility."""
    vol: pd.DataFrame = returns.ewm(span=span, min_periods=span).std()
    if annualise:
        vol = vol * np.sqrt(252)
    return vol



#  Correlation

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-period pairwise correlation matrix."""
    return returns.corr()


def rolling_correlation(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 63,
) -> pd.Series:
    """Rolling pairwise correlation between two tickers."""
    return returns[ticker_a].rolling(window).corr(returns[ticker_b])


def rolling_correlation_to_market(
    returns: pd.DataFrame,
    market_col: str | None = None,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling correlation of every ticker to the equal-weight market return.

    If market_col is provided, use that column as the market proxy.
    Otherwise compute the equal-weight mean across all columns.
    """
    if market_col and market_col in returns.columns:
        market: pd.Series = returns[market_col]
    else:
        market = returns.mean(axis=1)

    # Vectorised: rolling corr of each column with market
    result: dict[str, pd.Series] = {}
    for col in returns.columns:
        result[col] = returns[col].rolling(window).corr(market)
    return pd.DataFrame(result, index=returns.index)



#  Drawdown analysis

def drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown from running peak for each column.

    Returns negative values (e.g. -0.15 = 15% below peak).
    Fully vectorised via cummax.
    """
    peak: pd.DataFrame = prices.cummax()
    dd: pd.DataFrame = (prices - peak) / peak
    return dd


def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Maximum drawdown per ticker (most negative value)."""
    return drawdown(prices).min()


def drawdown_duration(prices: pd.DataFrame) -> pd.Series:
    """
    Longest drawdown duration in trading days per ticker.

    Vectorised per-column using cumsum tricks.
    """
    dd: pd.DataFrame = drawdown(prices)
    durations: dict[str, int] = {}
    for col in dd.columns:
        in_dd: pd.Series = (dd[col] < 0).astype(int)
        # Group consecutive drawdown days using cumsum of recovery points
        recovery_points: pd.Series = (in_dd == 0).cumsum()
        if in_dd.sum() == 0:
            durations[col] = 0
        else:
            durations[col] = int(in_dd.groupby(recovery_points).sum().max())
    return pd.Series(durations)



#  Risk metrics

def sharpe_ratio(
    returns: pd.DataFrame,
    risk_free_annual: float = 0.04,
) -> pd.Series:
    """
    Annualised Sharpe ratio per ticker.

    Uses mean daily return and daily std, annualised.
    """
    daily_rf: float = risk_free_annual / 252
    excess: pd.DataFrame = returns - daily_rf
    annualised_return: pd.Series = excess.mean() * 252
    annualised_vol: pd.Series = returns.std() * np.sqrt(252)
    return annualised_return / annualised_vol


def sortino_ratio(
    returns: pd.DataFrame,
    risk_free_annual: float = 0.04,
) -> pd.Series:
    """
    Annualised Sortino ratio per ticker.

    Uses downside deviation instead of total std.
    """
    daily_rf: float = risk_free_annual / 252
    excess: pd.DataFrame = returns - daily_rf
    annualised_return: pd.Series = excess.mean() * 252
    # Downside deviation: std of negative excess returns only
    downside: pd.DataFrame = excess.clip(upper=0)
    downside_std: pd.Series = downside.std() * np.sqrt(252)
    return annualised_return / downside_std


def beta(
    returns: pd.DataFrame,
    market: pd.Series | None = None,
) -> pd.Series:
    """
    Beta of each ticker relative to the equal-weight market.

    β = Cov(r_i, r_m) / Var(r_m), computed vectorised.
    """
    if market is None:
        market = returns.mean(axis=1)
    market_var: float = float(market.var())
    if market_var == 0:
        return pd.Series(np.nan, index=returns.columns)
    covariances: pd.Series = returns.apply(lambda col: col.cov(market))
    return covariances / market_var


def value_at_risk(
    returns: pd.DataFrame,
    confidence: float = 0.05,
) -> pd.Series:
    """Historical Value at Risk at given confidence level."""
    return returns.quantile(confidence)


def cvar(
    returns: pd.DataFrame,
    confidence: float = 0.05,
) -> pd.Series:
    """Conditional VaR (Expected Shortfall) — mean of returns below VaR."""
    var: pd.Series = value_at_risk(returns, confidence)
    # Vectorised: mask returns below VaR per column, take mean
    masked: pd.DataFrame = returns.where(returns <= var)
    return masked.mean()

#  Summary statistics

def summary_stats(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Comprehensive per-ticker summary.

    Returns a DataFrame with one row per ticker and columns for
    annualised return, vol, Sharpe, Sortino, max drawdown, VaR, CVaR, beta.
    """
    market: pd.Series = returns.mean(axis=1)
    stats: pd.DataFrame = pd.DataFrame(
        {
            "annual_return": returns.mean() * 252,
            "annual_vol": returns.std() * np.sqrt(252),
            "sharpe": sharpe_ratio(returns),
            "sortino": sortino_ratio(returns),
            "max_drawdown": max_drawdown(prices),
            "var_5pct": value_at_risk(returns),
            "cvar_5pct": cvar(returns),
            "beta": beta(returns, market),
        }
    )
    return stats.round(4)

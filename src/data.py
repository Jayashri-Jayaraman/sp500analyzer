"""
Data layer — load, fetch, and generate S&P 500 price data.

All operations are vectorised.  No loops touch row-level data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
#  Sector mapping (representative tickers)
# ──────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "BRK.B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "SPG": "Real Estate",
    "LIN": "Materials",
    "APD": "Materials",
    "ECL": "Materials",
    "SHW": "Materials",
    "NEM": "Materials",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "RTX": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "CAT": "Industrials",
    "BA": "Industrials",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "NFLX": "Communication Services",
}

TICKERS: list[str] = list(SECTOR_MAP.keys())
SECTORS: list[str] = sorted(set(SECTOR_MAP.values()))


def get_sector_series() -> pd.Series:
    """Return a ticker→sector mapping as a Series."""
    return pd.Series(SECTOR_MAP, name="sector")


# ──────────────────────────────────────────────
#  Data I/O
# ──────────────────────────────────────────────
def load_prices(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV of daily close prices.

    Expected shape: DatetimeIndex rows × ticker columns.
    """
    df: pd.DataFrame = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.astype(np.float64)
    return df


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    """Write prices DataFrame to CSV."""
    df.to_csv(path)


# ──────────────────────────────────────────────
#  Synthetic data generator (vectorised)
# ──────────────────────────────────────────────
def generate_synthetic_prices(
    tickers: list[str] | None = None,
    years: int = 10,
    start: str = "2016-01-04",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic correlated daily close prices.

    Uses Cholesky decomposition on a sector-based correlation structure
    so that same-sector stocks co-move.  Entirely vectorised — the only
    loop is over sectors (11 iterations), not over rows or tickers.

    Parameters
    ----------
    tickers : list of ticker strings (default: all 55 in SECTOR_MAP)
    years   : number of years of data
    start   : first trading date
    seed    : random seed for reproducibility

    Returns
    -------
    DataFrame with DatetimeIndex and one column per ticker.
    """
    if tickers is None:
        tickers = TICKERS

    rng = np.random.default_rng(seed)
    dates: pd.DatetimeIndex = pd.bdate_range(start=start, periods=252 * years)
    n_days: int = len(dates)
    n_tickers: int = len(tickers)

    # Build sector-aware correlation matrix
    sectors_arr: np.ndarray = np.array([SECTOR_MAP.get(t, "Other") for t in tickers])
    corr: np.ndarray = np.full((n_tickers, n_tickers), 0.30)  # baseline
    for sector in np.unique(sectors_arr):
        mask: np.ndarray = sectors_arr == sector
        corr[np.ix_(mask, mask)] = 0.65  # intra-sector correlation
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition for correlated normals
    chol: np.ndarray = np.linalg.cholesky(corr)
    raw_normals: np.ndarray = rng.standard_normal((n_days, n_tickers))
    correlated: np.ndarray = raw_normals @ chol.T

    # Per-ticker drift and vol (realistic ranges)
    annual_drift: np.ndarray = rng.uniform(0.02, 0.15, n_tickers)
    annual_vol: np.ndarray = rng.uniform(0.15, 0.45, n_tickers)
    daily_drift: np.ndarray = annual_drift / 252
    daily_vol: np.ndarray = annual_vol / np.sqrt(252)

    # Geometric Brownian Motion — fully vectorised
    log_returns: np.ndarray = daily_drift + daily_vol * correlated
    cum_returns: np.ndarray = np.cumsum(log_returns, axis=0)

    # Starting prices between $20 and $500
    start_prices: np.ndarray = rng.uniform(20, 500, n_tickers)
    prices: np.ndarray = start_prices * np.exp(cum_returns)

    return pd.DataFrame(prices, index=dates, columns=tickers).round(2)


# ──────────────────────────────────────────────
#  Returns computation
# ──────────────────────────────────────────────
def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from a price DataFrame.

    Parameters
    ----------
    method : 'log' for log returns, 'simple' for arithmetic returns.
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        returns: pd.DataFrame = prices.pct_change().dropna()
        return returns
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

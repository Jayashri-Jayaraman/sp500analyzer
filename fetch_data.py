"""
Fetch real S&P 500 price data via yfinance.

Usage:
    pip install yfinance
    python fetch_data.py

Writes data/prices.csv — the dashboard auto-detects this file.
"""

from __future__ import annotations

from pathlib import Path

import yfinance as yf

from src.data import TICKERS, save_prices


def main() -> None:
    print(f"Fetching 10 years of daily data for {len(TICKERS)} tickers...")
    data = yf.download(
        tickers=TICKERS,
        period="10y",
        interval="1d",
        auto_adjust=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns; extract 'Close'
    prices = data["Close"]
    prices = prices.dropna(how="all")

    # Drop tickers with >20% missing days
    threshold = len(prices) * 0.8
    prices = prices.dropna(axis=1, thresh=int(threshold))
    prices = prices.ffill().bfill()

    out_path = Path("data/prices.csv")
    out_path.parent.mkdir(exist_ok=True)
    save_prices(prices, out_path)

    print(f"Saved {prices.shape[0]} days × {prices.shape[1]} tickers to {out_path}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


if __name__ == "__main__":
    main()

# S&P 500 Vectorised Analysis Engine

A production-grade quantitative analysis engine for S&P 500 constituents. 55 tickers across 11 sectors, 10 years of daily data, fully vectorised with Pandas and NumPy — zero Python-level loops over rows. Interactive Streamlit dashboard with Plotly charts. Full test suite with GitHub Actions CI.

## What It Computes

**Per-ticker metrics:** annualised return, annualised volatility (rolling + EWM), Sharpe ratio, Sortino ratio, beta, Value at Risk (5%), Conditional VaR, maximum drawdown, drawdown duration.

**Cross-asset:** full pairwise correlation matrix, rolling pairwise correlations, extreme correlation pairs (most/least correlated).

**Sector-level:** equal-weight sector returns, sector cumulative performance, sector rolling volatility, sector correlation heatmap, sector momentum (63-day), sector relative strength, price-weighted sector allocation over time.

## Architecture

```
src/
├── data.py       # Data I/O, synthetic generation (Cholesky-correlated GBM)
├── analysis.py   # Rolling vol, correlations, drawdowns, risk metrics
└── sectors.py    # Sector aggregations, momentum, rotation signals
```

Every function is `DataFrame → DataFrame` (or `→ Series`). No `iterrows()`, no `apply(lambda)`, no Python-level loops over data rows. The only loops in the codebase iterate over column names (11 sectors, 55 tickers) — never over the ~2,500 daily observations.

Key vectorisation patterns used:
- `pd.DataFrame.rolling().std()` for rolling volatility
- `pd.DataFrame.cummax()` for drawdown computation
- `np.linalg.cholesky()` for correlated return generation
- `pd.DataFrame.T.groupby().mean().T` for sector aggregation on column axis
- `pd.Series.rolling().corr()` for rolling pairwise correlation
- `pd.DataFrame.where()` for conditional VaR masking

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/sp500-analysis-engine.git
cd sp500-analysis-engine
pip install -r requirements.txt

# Run dashboard (uses synthetic data by default)
streamlit run dashboard.py

# Optional: fetch real data from Yahoo Finance
pip install yfinance
python fetch_data.py
streamlit run dashboard.py  # now uses real data
```

## Dashboard Pages

- **Overview** — growth-of-$1 chart, top-level market metrics, per-ticker summary table with Sharpe gradient
- **Volatility** — rolling and EWM volatility with adjustable window, current vol distribution bar chart
- **Correlations** — full ticker heatmap, sector heatmap, extreme correlation pairs table
- **Sectors** — cumulative returns, performance table, rolling vol, stacked area allocation, momentum ranking
- **Risk Metrics** — drawdown chart, risk summary table, return distribution histograms, risk-return scatter plot

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest -v

# Format
black src/ tests/ dashboard.py

# Type check
mypy src/

# All three (mirrors CI)
pytest -v && black --check src/ tests/ dashboard.py && mypy src/
```

## CI Pipeline

GitHub Actions runs on every push and PR against `main`:
1. Matrix: Python 3.10, 3.11, 3.12
2. `black --check` — formatting gate
3. `mypy src/` — type checking gate
4. `pytest -v` — full test suite
5. Coverage report (term-missing)

## Data

**Synthetic mode (default):** Generates 10 years of realistic daily prices for 55 tickers using geometric Brownian motion with sector-correlated returns (Cholesky decomposition). Deterministic with `seed=42`. No internet connection required.

**Real data mode:** Run `python fetch_data.py` to download actual prices from Yahoo Finance. The dashboard auto-detects `data/prices.csv` and switches to real data.

## Design Decisions

**Why synthetic data as default?** Reproducibility and zero external dependencies for tests and CI. Real data requires `yfinance` and network access, which breaks in CI environments. Synthetic data is deterministic and fast.

**Why not market-cap weighting?** We don't have shares outstanding in the dataset. Price-weighted sector allocation is the honest alternative — it's transparent about what it measures.

**Why `apply(lambda)` in `beta()`?** That single `apply` iterates over 55 columns, not 2,500 rows. The column-level covariance calculation inside it (`col.cov(market)`) is itself vectorised over all rows. This is the correct granularity for vectorisation — eliminate row-level loops, not column-level iteration.

## Requirements

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- plotly >= 5.18
- streamlit >= 1.30
- Dev: black, mypy, pytest, pandas-stubs
- Optional: yfinance (for real data)

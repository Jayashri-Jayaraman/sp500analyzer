"""
Streamlit dashboard — S&P 500 vectorised analysis engine.

Run: streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import (
    correlation_matrix,
    drawdown,
    ewm_volatility,
    max_drawdown,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    summary_stats,
    value_at_risk,
)
from src.data import (
    SECTOR_MAP,
    SECTORS,
    TICKERS,
    compute_returns,
    generate_synthetic_prices,
    get_sector_series,
    load_prices,
)
from src.sectors import (
    sector_correlation_matrix,
    sector_cumulative_returns,
    sector_momentum,
    sector_performance_summary,
    sector_returns,
    sector_rolling_volatility,
    sector_weight_by_price,
)

#  Page config

st.set_page_config(
    page_title="S&P 500 Analysis Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom styling

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #0f3460;
    }
    .metric-card h3 { color: #e94560; margin: 0 0 8px 0; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-card .value { color: #eee; font-size: 1.8rem;
        font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; }
    div[data-testid="stSidebar"] { background: #0a0a1a; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_TEMPLATE = "plotly_dark"
COLOR_PALETTE: list[str] = [
    "#e94560",
    "#0f3460",
    "#533483",
    "#16c79a",
    "#f5a623",
    "#4ecdc4",
    "#ff6b6b",
    "#c44dff",
    "#45b7d1",
    "#96ceb4",
    "#feca57",
]


#  Data loading (cached)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or generate price data, compute returns."""
    data_path: Path = Path("data/prices.csv")
    if data_path.exists():
        prices = load_prices(data_path)
    else:
        prices = generate_synthetic_prices()
    returns = compute_returns(prices, method="log")
    return prices, returns


prices, returns = load_data()
sector_map: pd.Series = get_sector_series()


#  Helper: safe defaults for multiselect


def safe_defaults(candidates: list[str], available: list[str]) -> list[str]:
    """Filter default tickers to only those present in the data."""
    result = [t for t in candidates if t in available]
    if not result and available:
        return available[:3]
    return result


available_tickers: list[str] = list(prices.columns)


#  Sidebar

st.sidebar.title("📊 S&P 500 Engine")
st.sidebar.markdown("---")

page: str = st.sidebar.radio(
    "Navigate",
    ["Overview", "Volatility", "Correlations", "Sectors", "Risk Metrics"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{len(prices)}** trading days · **{len(prices.columns)}** tickers · "
    f"**{prices.index[0].strftime('%Y-%m-%d')}** to **{prices.index[-1].strftime('%Y-%m-%d')}**"
)


#  Helper: metric card


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f"<h3>{label}</h3>"
        f'<div class="value">{value}</div>'
        f"</div>"
    )


#  Helper: safe style formatting (no matplotlib needed)


def format_stats_table(stats: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Format the summary stats table without background_gradient to avoid matplotlib dependency."""
    return stats.style.format(
        {
            "annual_return": "{:.2%}",
            "annual_vol": "{:.2%}",
            "sharpe": "{:.2f}",
            "sortino": "{:.2f}",
            "max_drawdown": "{:.2%}",
            "var_5pct": "{:.4f}",
            "cvar_5pct": "{:.4f}",
            "beta": "{:.2f}",
        }
    )


def try_gradient(styler: pd.io.formats.style.Styler, column: str) -> pd.io.formats.style.Styler:
    """Apply background_gradient if matplotlib is available, otherwise skip."""
    try:
        return styler.background_gradient(subset=[column], cmap="RdYlGn")
    except ImportError:
        return styler


#  PAGE: Overview

if page == "Overview":
    st.title("Market Overview")
    st.markdown("10 years of S&P 500 constituent data · fully vectorised analysis")

    # Top-level metrics
    market_return: float = float(returns.mean(axis=1).sum()) * 100
    market_vol: float = float(returns.mean(axis=1).std() * np.sqrt(252)) * 100
    avg_sharpe: float = float(sharpe_ratio(returns).mean())
    avg_mdd: float = float(max_drawdown(prices).mean()) * 100

    cols = st.columns(4)
    cols[0].markdown(
        metric_card("Cumulative Return", f"{market_return:.1f}%"), unsafe_allow_html=True
    )
    cols[1].markdown(metric_card("Annualised Vol", f"{market_vol:.1f}%"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Avg Sharpe", f"{avg_sharpe:.2f}"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Avg Max Drawdown", f"{avg_mdd:.1f}%"), unsafe_allow_html=True)

    st.markdown("---")

    # Normalised price chart (growth of $1)
    normalised: pd.DataFrame = prices / prices.iloc[0]
    selected_tickers: list[str] = st.multiselect(
        "Select tickers to plot",
        options=available_tickers,
        default=safe_defaults(["AAPL", "MSFT", "JPM", "XOM", "UNH"], available_tickers),
    )

    if selected_tickers:
        fig = px.line(
            normalised[selected_tickers],
            template=PLOT_TEMPLATE,
            title="Growth of $1",
            labels={"value": "Growth", "variable": "Ticker"},
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig.update_layout(
            height=500,
            legend=dict(orientation="h", y=-0.15),
            xaxis_title="",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats table
    st.subheader("Per-Ticker Summary")
    stats: pd.DataFrame = summary_stats(returns, prices)
    stats["sector"] = stats.index.map(SECTOR_MAP)
    stats = stats[["sector"] + [c for c in stats.columns if c != "sector"]]

    styled = format_stats_table(stats)
    styled = try_gradient(styled, "sharpe")
    st.dataframe(styled, use_container_width=True, height=600)


#  PAGE: Volatility

elif page == "Volatility":
    st.title("Volatility Analysis")

    vol_col1, vol_col2 = st.columns(2)
    with vol_col1:
        vol_window: int = st.slider("Rolling window (days)", 5, 126, 21, step=1)
    with vol_col2:
        vol_tickers: list[str] = st.multiselect(
            "Tickers",
            available_tickers,
            default=safe_defaults(["AAPL", "MSFT", "NVDA", "XOM", "JPM"], available_tickers),
            key="vol_tickers",
        )

    if vol_tickers:
        # Rolling vol
        rvol: pd.DataFrame = rolling_volatility(returns[vol_tickers], window=vol_window)

        fig_vol = px.line(
            rvol,
            template=PLOT_TEMPLATE,
            title=f"{vol_window}-Day Rolling Annualised Volatility",
            labels={"value": "Volatility", "variable": "Ticker"},
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_vol.update_layout(height=450, hovermode="x unified", yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

        # EWM vol comparison
        ewm_vol: pd.DataFrame = ewm_volatility(returns[vol_tickers], span=vol_window)

        fig_ewm = px.line(
            ewm_vol,
            template=PLOT_TEMPLATE,
            title=f"Exponentially Weighted Volatility (span={vol_window})",
            labels={"value": "Volatility", "variable": "Ticker"},
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_ewm.update_layout(height=450, hovermode="x unified", yaxis_tickformat=".0%")
        st.plotly_chart(fig_ewm, use_container_width=True)

    # Vol distribution
    st.subheader("Volatility Distribution (Current)")
    current_vol: pd.Series = rolling_volatility(returns, window=vol_window).iloc[-1].dropna()
    current_vol = current_vol.sort_values(ascending=False)

    fig_bar = px.bar(
        x=current_vol.index,
        y=current_vol.values,
        template=PLOT_TEMPLATE,
        title=f"Current {vol_window}-Day Annualised Volatility by Ticker",
        color=current_vol.values,
        color_continuous_scale="Reds",
    )
    fig_bar.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="Annualised Vol",
        yaxis_tickformat=".0%",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


#  PAGE: Correlations

elif page == "Correlations":
    st.title("Correlation Analysis")

    tab1, tab2 = st.tabs(["Ticker Heatmap", "Sector Heatmap"])

    with tab1:
        corr: pd.DataFrame = correlation_matrix(returns)

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 7},
            )
        )
        fig_hm.update_layout(
            template=PLOT_TEMPLATE,
            title="Full-Period Ticker Correlation Matrix",
            height=800,
            width=900,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab2:
        sec_corr: pd.DataFrame = sector_correlation_matrix(returns)

        fig_sec = go.Figure(
            data=go.Heatmap(
                z=sec_corr.values,
                x=sec_corr.columns.tolist(),
                y=sec_corr.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=sec_corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )
        fig_sec.update_layout(
            template=PLOT_TEMPLATE,
            title="Sector Correlation Matrix",
            height=600,
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    # Highest / lowest correlations
    st.subheader("Extreme Correlations")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    flat: pd.Series = corr.where(mask).stack().sort_values()

    col_lo, col_hi = st.columns(2)
    with col_lo:
        st.markdown("**Lowest (most diversifying)**")
        bottom: pd.Series = flat.head(10)
        st.dataframe(
            pd.DataFrame(
                {"pair": [f"{a} / {b}" for a, b in bottom.index], "corr": bottom.values}
            ).style.format({"corr": "{:.3f}"})
        )
    with col_hi:
        st.markdown("**Highest (most co-moving)**")
        top: pd.Series = flat.tail(10).iloc[::-1]
        st.dataframe(
            pd.DataFrame(
                {"pair": [f"{a} / {b}" for a, b in top.index], "corr": top.values}
            ).style.format({"corr": "{:.3f}"})
        )


#  PAGE: Sectors

elif page == "Sectors":
    st.title("Sector Analysis")

    # Cumulative returns
    sec_cum: pd.DataFrame = sector_cumulative_returns(returns)
    fig_sec_cum = px.line(
        sec_cum,
        template=PLOT_TEMPLATE,
        title="Sector Cumulative Returns (Growth of $1)",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig_sec_cum.update_layout(
        height=500,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig_sec_cum, use_container_width=True)

    # Performance table
    st.subheader("Sector Performance Summary")
    sec_perf: pd.DataFrame = sector_performance_summary(returns, prices)
    sec_styled = sec_perf.sort_values("sharpe", ascending=False).style.format(
        {
            "annual_return": "{:.2%}",
            "annual_vol": "{:.2%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.2%}",
        }
    )
    sec_styled = try_gradient(sec_styled, "sharpe")
    st.dataframe(sec_styled, use_container_width=True)

    st.markdown("---")

    # Rolling vol by sector
    sec_rvol: pd.DataFrame = sector_rolling_volatility(returns, window=63)
    fig_sec_vol = px.line(
        sec_rvol,
        template=PLOT_TEMPLATE,
        title="63-Day Rolling Sector Volatility",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig_sec_vol.update_layout(height=450, hovermode="x unified", yaxis_tickformat=".0%")
    st.plotly_chart(fig_sec_vol, use_container_width=True)

    # Sector weight evolution
    st.subheader("Sector Weight (Price-Weighted)")
    weights: pd.DataFrame = sector_weight_by_price(prices)
    fig_area = px.area(
        weights,
        template=PLOT_TEMPLATE,
        title="Sector Allocation Over Time",
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig_area.update_layout(
        height=450,
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # Sector momentum
    st.subheader("Sector Momentum (63-Day)")
    sec_mom: pd.DataFrame = sector_momentum(returns, lookback=63)
    latest_mom: pd.Series = sec_mom.iloc[-1].sort_values(ascending=True)

    fig_mom = px.bar(
        x=latest_mom.values,
        y=latest_mom.index,
        orientation="h",
        template=PLOT_TEMPLATE,
        title="Current Sector Momentum (63-Day Cumulative Return)",
        color=latest_mom.values,
        color_continuous_scale="RdYlGn",
    )
    fig_mom.update_layout(height=400, xaxis_tickformat=".1%", yaxis_title="")
    st.plotly_chart(fig_mom, use_container_width=True)


#  PAGE: Risk Metrics
elif page == "Risk Metrics":
    st.title("Risk Metrics")

    risk_tickers: list[str] = st.multiselect(
        "Select tickers",
        available_tickers,
        default=safe_defaults(["AAPL", "TSLA", "JNJ", "XOM", "NVDA"], available_tickers),
        key="risk_tickers",
    )

    if risk_tickers:
        # Drawdown chart
        dd: pd.DataFrame = drawdown(prices[risk_tickers])
        fig_dd = px.area(
            dd,
            template=PLOT_TEMPLATE,
            title="Drawdown from Peak",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_dd.update_layout(
            height=400,
            yaxis_tickformat=".0%",
            hovermode="x unified",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Risk table
        st.subheader("Risk Summary")
        risk_stats: pd.DataFrame = summary_stats(returns[risk_tickers], prices[risk_tickers])
        risk_stats["sector"] = risk_stats.index.map(SECTOR_MAP)
        st.dataframe(
            format_stats_table(risk_stats),
            use_container_width=True,
        )

        # Return distribution
        st.subheader("Return Distributions")
        fig_hist = make_subplots(
            rows=1,
            cols=len(risk_tickers),
            subplot_titles=risk_tickers,
        )
        for i, ticker in enumerate(risk_tickers, 1):
            fig_hist.add_trace(
                go.Histogram(
                    x=returns[ticker].values,
                    nbinsx=80,
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    name=ticker,
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
        fig_hist.update_layout(
            template=PLOT_TEMPLATE,
            height=350,
            title_text="Daily Return Distributions",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Risk-return scatter
        st.subheader("Risk-Return Tradeoff")
        all_stats: pd.DataFrame = summary_stats(returns, prices)
        all_stats["sector"] = all_stats.index.map(SECTOR_MAP)
        all_stats["ticker"] = all_stats.index

        fig_scatter = px.scatter(
            all_stats,
            x="annual_vol",
            y="annual_return",
            color="sector",
            text="ticker",
            template=PLOT_TEMPLATE,
            title="Annualised Return vs Volatility",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=8)
        fig_scatter.update_layout(
            height=550,
            xaxis_title="Annualised Volatility",
            yaxis_title="Annualised Return",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

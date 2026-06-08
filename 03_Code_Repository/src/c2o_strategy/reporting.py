"""Output tables, figures and HTML reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig
from c2o_strategy.metrics import max_drawdown, sharpe_ratio


def compute_equal_weight_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the equal-weighted overnight/intraday/total stylised-fact series."""

    mask = panel["in_universe"] & panel["r_overnight"].notna() & panel["r_intraday"].notna()
    daily = (
        panel.loc[mask]
        .groupby("date")
        .agg(
            overnight=("r_overnight", "mean"),
            intraday=("r_intraday", "mean"),
            close_to_close=("r_close_close", "mean"),
            names=("instrument_id", "nunique"),
        )
        .reset_index()
    )
    return daily


def plot_equal_weight_decomposition(decomposition: pd.DataFrame, path: Path) -> None:
    """Plot cumulative growth of overnight, intraday and close-to-close streams."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column, label in [
        ("overnight", "Overnight"),
        ("intraday", "Intraday"),
        ("close_to_close", "Close-to-close"),
    ]:
        growth = (1.0 + decomposition[column].fillna(0.0)).cumprod()
        ax.plot(decomposition["date"], growth, label=label, linewidth=1.6)
    ax.set_title("Equal-weight universe return decomposition")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_strategy_cumulative(daily_by_aum: dict[float, pd.DataFrame], path: Path) -> None:
    """Plot cumulative net strategy returns for all required AUM levels."""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for aum, daily in daily_by_aum.items():
        growth = (1.0 + daily["net_return"].fillna(0.0)).cumprod()
        ax.plot(daily["date"], growth, label=f"${aum/1_000_000:.0f}M", linewidth=1.5)
    ax.set_title("C2O strategy net cumulative return")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Portfolio AUM")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ic(ic_daily: pd.DataFrame, path: Path) -> None:
    """Plot rolling mean IC."""

    if ic_daily.empty:
        return
    data = ic_daily.sort_values("date").copy()
    data["rolling_63d_ic"] = data["ic"].rolling(63, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["date"], data["rolling_63d_ic"], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("63-day rolling Information Coefficient")
    ax.set_ylabel("Spearman IC")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_tearsheet(
    returns: pd.Series, benchmark: pd.Series, output_path: Path, title: str
) -> None:
    """Generate QuantStats HTML, falling back to a compact HTML if unavailable."""

    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index().dropna()
    benchmark = benchmark.copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark = benchmark.sort_index().dropna()

    try:
        import quantstats as qs

        qs.extend_pandas()
        qs.reports.html(returns, benchmark=benchmark, output=str(output_path), title=title)
        return
    except Exception as exc:  # pragma: no cover - exercised only without quantstats
        _write_fallback_html(returns, benchmark, output_path, title, exc)


def _write_fallback_html(
    returns: pd.Series, benchmark: pd.Series, output_path: Path, title: str, exc: Exception
) -> None:
    joined = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    strategy_growth = (1.0 + joined["strategy"]).cumprod()
    benchmark_growth = (1.0 + joined["benchmark"]).cumprod()
    drawdown = strategy_growth / strategy_growth.cummax() - 1.0

    plot_path = output_path.with_suffix(".fallback.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(strategy_growth.index, strategy_growth, label="Strategy")
    axes[0].plot(benchmark_growth.index, benchmark_growth, label="SP500_TR")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p><strong>QuantStats was unavailable or failed:</strong> {exc}</p>
<p>This fallback is not a substitute for the required QuantStats artefact, but
it keeps the reproduction command auditable until dependencies are installed.</p>
<ul>
<li>Strategy Sharpe: {sharpe_ratio(joined['strategy']):.3f}</li>
<li>Benchmark Sharpe: {sharpe_ratio(joined['benchmark']):.3f}</li>
<li>Strategy max drawdown: {max_drawdown(joined['strategy']):.2%}</li>
<li>Benchmark max drawdown: {max_drawdown(joined['benchmark']):.2%}</li>
</ul>
<img src="{plot_path.name}" alt="fallback cumulative return and drawdown">
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_summary_markdown(config: StrategyConfig, performance: pd.DataFrame, output_path: Path) -> None:
    """Write a compact run summary for report traceability."""

    lines = [
        "# C2O Strategy Reproduction Summary",
        "",
        f"- Start date: `{config.start_date}`",
        f"- Data cutoff: `{config.cutoff}`",
        f"- Development cutoff: `{config.development_cutoff}`",
        f"- Universe size: `{config.universe_size}`",
        f"- Minimum price: `${config.min_price:.2f}`",
        f"- ADV20 floor: `${config.min_adv20:,.0f}`",
        f"- Volatility band: `{config.vol_floor:.0%}` to `{config.vol_cap:.0%}` annualised",
        f"- Earnings exclusion window: `+/- {config.earnings_window_days}` trading day(s)",
        f"- Participation cap: `{config.participation_cap:.1%}` of ADV20",
        f"- Basket fraction: `{config.basket_fraction:.0%}` per side",
        "",
        "## Headline Metrics",
        "",
        performance.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")

"""Data loading, return construction and point-in-time universe logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig


PRICE_COLUMNS = [
    "ticker",
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "market_cap",
    "status",
]

AUX_COLUMNS = [
    "stock_id",
    "date",
    "piot_norm",
    "dsi",
    "dtcn",
    "ddtcn",
    "short_interest",
    "asset_turnover_ratio",
    "ev_to_ebit",
    "gross_profit_margin",
    "net_debt_to_equity",
    "price_to_book",
    "epsp",
    "epsf",
    "reps1",
    "repsf4",
    "sue",
    "inesp",
    "inesn",
    "reps41",
    "repsfs",
    "repsfl",
    "nspc5",
    "deps",
    "gics_sector",
    "gics_group",
    "gics_industry",
    "gics_subindustry",
    "industry_return",
]

CHEAPNESS_COLUMNS = [
    "instrument_id",
    "date",
    "valuation_score",
    "quality_score",
    "health_score",
    "momentum_score",
    "final_score",
    "final_score_clean",
    "score_velocity",
    "score_acceleration",
    "regime_break",
    "value_trap",
]


@dataclass(frozen=True)
class PreparedData:
    """Container for the prepared panel and audit tables."""

    panel: pd.DataFrame
    universe: pd.DataFrame
    universe_summary: pd.DataFrame
    reconciliation_summary: pd.DataFrame


def load_prices(config: StrategyConfig) -> pd.DataFrame:
    """Load daily prices up to the configured cutoff and adjust auction prices."""

    path = config.data_dir / "prices.parquet"
    prices = pd.read_parquet(path, columns=PRICE_COLUMNS)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"].le(config.cutoff_ts)].copy()
    prices = prices.sort_values(["instrument_id", "date"], kind="mergesort")

    adj_factor = prices["adjusted_close"] / prices["close"]
    adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan)
    for column in ["open", "high", "low", "close"]:
        prices[f"adj_{column}"] = prices[column] * adj_factor
    prices["adj_close"] = prices["adjusted_close"]
    prices["dollar_volume"] = prices["adj_close"] * prices["volume"]
    prices["history_count"] = prices.groupby("instrument_id").cumcount() + 1
    return prices


def build_return_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Construct the return objects required by Section 2.1 of the brief."""

    panel = prices[
        [
            "ticker",
            "instrument_id",
            "date",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
            "dollar_volume",
            "market_cap",
            "history_count",
        ]
    ].copy()
    grouped = panel.groupby("instrument_id", sort=False)
    panel["lag_adj_close"] = grouped["adj_close"].shift(1)
    panel["lead_adj_open"] = grouped["adj_open"].shift(-1)
    panel["next_date"] = grouped["date"].shift(-1)

    panel["r_overnight"] = panel["adj_open"] / panel["lag_adj_close"] - 1.0
    panel["r_intraday"] = panel["adj_close"] / panel["adj_open"] - 1.0
    panel["r_close_close"] = panel["adj_close"] / panel["lag_adj_close"] - 1.0
    panel["overnight_next"] = panel["lead_adj_open"] / panel["adj_close"] - 1.0
    panel["return_identity_residual"] = (
        (1.0 + panel["r_overnight"]) * (1.0 + panel["r_intraday"])
        - 1.0
        - panel["r_close_close"]
    )

    panel["price_lag1"] = grouped["adj_close"].shift(1)
    panel["market_cap_lag1"] = grouped["market_cap"].shift(1)
    panel["dollar_volume_lag1"] = grouped["dollar_volume"].shift(1)

    panel["adv20"] = grouped["dollar_volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean().shift(1)
    )
    panel["vol20"] = grouped["r_close_close"].transform(
        lambda x: x.rolling(20, min_periods=10).std().shift(1) * np.sqrt(252.0)
    )
    panel["vol60"] = grouped["r_close_close"].transform(
        lambda x: x.rolling(60, min_periods=30).std().shift(1) * np.sqrt(252.0)
    )
    abs_ret_over_dv = panel["r_close_close"].abs() / panel["dollar_volume"].replace(0, np.nan)
    panel["amihud20"] = abs_ret_over_dv.groupby(panel["instrument_id"]).transform(
        lambda x: x.rolling(20, min_periods=10).mean().shift(1)
    )
    return panel


def build_yearly_universe(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Freeze the top-N market-cap universe on the first trading day of each year."""

    rows: list[pd.DataFrame] = []
    years = range(config.start_ts.year, config.cutoff_ts.year + 1)
    for year in years:
        jan1 = pd.Timestamp(year=year, month=1, day=1)
        candidates = prices.loc[prices["date"].lt(jan1)]
        if candidates.empty:
            continue
        ref_date = candidates["date"].max()
        ref = candidates.loc[
            candidates["date"].eq(ref_date)
            & candidates["market_cap"].notna()
            & candidates["history_count"].ge(config.min_history_days)
        ].copy()
        ref = ref.sort_values("market_cap", ascending=False).head(config.universe_size)
        ref = ref[["instrument_id", "ticker", "market_cap", "history_count"]]
        ref["year"] = year
        ref["universe_ref_date"] = ref_date
        ref = ref.rename(columns={"market_cap": "year_start_market_cap"})
        rows.append(ref)
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument_id",
                "ticker",
                "year_start_market_cap",
                "history_count",
                "year",
                "universe_ref_date",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def merge_auxiliary_features(panel: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Merge daily public feature files into the price panel."""

    aux_path = config.data_dir / "all_data.parquet"
    if aux_path.exists():
        aux = pd.read_parquet(aux_path, columns=AUX_COLUMNS)
        aux = aux.rename(columns={"stock_id": "instrument_id"})
        aux["date"] = pd.to_datetime(aux["date"])
        aux = aux.loc[aux["date"].between(config.start_ts, config.cutoff_ts)]
        panel = panel.merge(aux, on=["instrument_id", "date"], how="left")

    cheap_path = config.data_dir / "cheapness_scores.parquet"
    if cheap_path.exists():
        cheap = pd.read_parquet(cheap_path, columns=CHEAPNESS_COLUMNS)
        cheap["date"] = pd.to_datetime(cheap["date"])
        cheap = cheap.loc[cheap["date"].between(config.start_ts, config.cutoff_ts)]
        cheap = cheap.drop_duplicates(["instrument_id", "date"])
        panel = panel.merge(cheap, on=["instrument_id", "date"], how="left")

    regime_path = config.data_dir / "regime.parquet"
    if regime_path.exists():
        regime = pd.read_parquet(regime_path)
        regime["date"] = pd.to_datetime(regime["date"])
        regime = regime.loc[regime["date"].between(config.start_ts, config.cutoff_ts)]
        regime = regime[["date", "regime"]].drop_duplicates("date")
        panel = panel.merge(regime, on="date", how="left")
    else:
        panel["regime"] = np.nan

    return panel


def build_earnings_window_flags(
    config: StrategyConfig, trading_dates: pd.Series
) -> pd.DataFrame:
    """Return decision-date flags around earnings, respecting the provided timing date."""

    path = config.data_dir / "earnings_calendar.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["instrument_id", "date", "earnings_window"])

    calendar = pd.read_parquet(path)
    calendar = calendar.rename(columns={"stock_id": "instrument_id"})
    calendar["strat_trading_date"] = pd.to_datetime(calendar["strat_trading_date"])
    calendar = calendar.loc[calendar["strat_trading_date"].notna()].copy()

    dates = pd.Index(pd.to_datetime(trading_dates.drop_duplicates()).sort_values())
    date_to_position = pd.Series(np.arange(len(dates)), index=dates)
    event_positions = calendar["strat_trading_date"].map(date_to_position)
    calendar = calendar.loc[event_positions.notna()].copy()
    calendar["date_position"] = event_positions.dropna().astype(int).to_numpy()

    rows: list[pd.DataFrame] = []
    window = config.earnings_window_days
    for offset in range(-window, window + 1):
        pos = calendar["date_position"] + offset
        valid = pos.between(0, len(dates) - 1)
        if not valid.any():
            continue
        chunk = calendar.loc[valid, ["instrument_id"]].copy()
        chunk["date"] = dates.take(pos.loc[valid].to_numpy())
        rows.append(chunk)

    if not rows:
        return pd.DataFrame(columns=["instrument_id", "date", "earnings_window"])

    flags = pd.concat(rows, ignore_index=True).drop_duplicates(["instrument_id", "date"])
    flags["earnings_window"] = True
    return flags


def attach_universe(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Mark whether each stock-day belongs to the frozen annual universe."""

    panel = panel.copy()
    panel["year"] = panel["date"].dt.year
    universe_key = universe[
        ["instrument_id", "year", "year_start_market_cap", "universe_ref_date"]
    ].copy()
    panel = panel.merge(universe_key, on=["instrument_id", "year"], how="left")
    panel["in_universe"] = panel["year_start_market_cap"].notna()
    return panel


def build_universe_summary(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Summarise yearly universe counts and mid-year exits."""

    if universe.empty:
        return pd.DataFrame()

    trading_dates = panel.groupby("year")["date"].max().rename("year_last_trading_date")
    last_seen = (
        panel.loc[panel["in_universe"]]
        .groupby(["year", "instrument_id"])["date"]
        .max()
        .rename("last_seen_date")
        .reset_index()
    )
    last_seen = last_seen.merge(trading_dates, on="year", how="left")
    last_seen["mid_year_exit"] = last_seen["last_seen_date"].lt(
        last_seen["year_last_trading_date"]
    )

    summary = (
        universe.groupby("year")
        .agg(
            universe_count=("instrument_id", "nunique"),
            ref_date=("universe_ref_date", "first"),
            median_year_start_mcap=("year_start_market_cap", "median"),
        )
        .reset_index()
    )
    exits = last_seen.groupby("year")["mid_year_exit"].sum().rename("mid_year_exits")
    summary = summary.merge(exits, on="year", how="left")
    summary["mid_year_exits"] = summary["mid_year_exits"].fillna(0).astype(int)
    return summary


def build_reconciliation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Audit the close/open/close return identity residual."""

    residual = panel["return_identity_residual"].replace([np.inf, -np.inf], np.nan).abs()
    tolerance = 1e-8
    valid = residual.notna()
    return pd.DataFrame(
        [
            {
                "tolerance": tolerance,
                "stock_days_checked": int(valid.sum()),
                "fail_count": int(residual.loc[valid].gt(tolerance).sum()),
                "fail_fraction": float(residual.loc[valid].gt(tolerance).mean()),
                "median_abs_residual": float(residual.loc[valid].median()),
                "p99_abs_residual": float(residual.loc[valid].quantile(0.99)),
                "max_abs_residual": float(residual.loc[valid].max()),
            }
        ]
    )


def prepare_data(config: StrategyConfig) -> PreparedData:
    """Load all raw inputs and return an auditable daily panel."""

    prices = load_prices(config)
    full_panel = build_return_panel(prices)
    universe = build_yearly_universe(prices, config)
    full_panel = attach_universe(full_panel, universe)

    panel = full_panel.loc[
        full_panel["date"].between(config.start_ts, config.cutoff_ts)
    ].copy()
    panel = merge_auxiliary_features(panel, config)

    earnings_flags = build_earnings_window_flags(config, full_panel["date"])
    panel = panel.merge(earnings_flags, on=["instrument_id", "date"], how="left")
    panel["earnings_window"] = panel["earnings_window"].eq(True)

    reconciliation_summary = build_reconciliation_summary(panel)
    universe_summary = build_universe_summary(panel, universe)
    return PreparedData(
        panel=panel,
        universe=universe,
        universe_summary=universe_summary,
        reconciliation_summary=reconciliation_summary,
    )


def load_benchmark_returns(config: StrategyConfig) -> pd.Series:
    """Load S&P 500 Total Return benchmark close-to-close returns."""

    path = config.data_dir / "sp500_tr.parquet"
    benchmark = pd.read_parquet(path)
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.loc[benchmark["date"].between(config.start_ts, config.cutoff_ts)]
    benchmark = benchmark.sort_values("date")
    returns = benchmark.set_index("date")["adjusted_close"].pct_change().dropna()
    returns.name = "SP500_TR"
    return returns

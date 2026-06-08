"""Controlled strategy improvement experiments for the C2O pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from c2o_strategy.alpha import _prior_weights
from c2o_strategy.config import StrategyConfig
from c2o_strategy.data import prepare_data
from c2o_strategy.features import (
    add_cross_sectional_ranks,
    add_point_in_time_features,
    apply_eligibility_filters,
)
from c2o_strategy.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
)
from c2o_strategy.portfolio import COMMISSION_ROUND_TRIP, SLIPPAGE_ROUND_TRIP


BASELINE_AUM = 250_000_000.0
DEFAULT_ALPHA_RETURN_SLOPE = 0.01
MIN_TRAINING_OBSERVATIONS = 20_000
TARGET_COLUMN = "experiment_target"
CACHE_VERSION = "v2"


@dataclass(frozen=True)
class ExperimentSpec:
    """One controlled experiment configuration."""

    experiment_group: str
    experiment_name: str
    long_fraction: float = 0.03
    short_fraction: float = 0.03
    weighting_scheme: str = "equal"
    ranking_scheme: str = "raw"
    short_treatment: str = "tiered_borrow"
    target_transform: str = "rank"
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C2O strategy improvement experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--development-cutoff", default="2024-12-31")
    return parser.parse_args()


def experiment_specs() -> list[ExperimentSpec]:
    """Return the full logged experiment grid requested by the brief."""

    return [
        ExperimentSpec("A_basket_size", "top_bottom_3pct", notes="Current baseline basket."),
        ExperimentSpec("A_basket_size", "top_bottom_5pct", 0.05, 0.05),
        ExperimentSpec("A_basket_size", "top_bottom_7_5pct", 0.075, 0.075),
        ExperimentSpec("A_basket_size", "top_bottom_10pct", 0.10, 0.10),
        ExperimentSpec("A_basket_size", "long_3pct_short_5pct", 0.03, 0.05),
        ExperimentSpec("A_basket_size", "long_5pct_short_10pct", 0.05, 0.10),
        ExperimentSpec("B_weighting", "equal_weight_baseline", notes="Current baseline weighting."),
        ExperimentSpec("B_weighting", "volatility_weighted", weighting_scheme="volatility"),
        ExperimentSpec("B_weighting", "score_weighted", weighting_scheme="score"),
        ExperimentSpec("B_weighting", "score_divided_by_volatility", weighting_scheme="score_div_vol"),
        ExperimentSpec(
            "B_weighting",
            "score_times_liquidity",
            weighting_scheme="score_times_liquidity",
            notes="Liquidity preference is capped by the same 5% ADV participation rule.",
        ),
        ExperimentSpec("C_cost_aware_ranking", "baseline_raw_score", notes="Current raw score ranking."),
        ExperimentSpec(
            "C_cost_aware_ranking",
            "alpha_minus_estimated_round_trip_cost",
            ranking_scheme="cost_adjusted",
            notes="Uses walk-forward calibrated alpha return minus fixed cost and daily liquidity-impact proxy.",
        ),
        ExperimentSpec(
            "C_cost_aware_ranking",
            "short_alpha_minus_borrow_and_liquidity_penalty",
            ranking_scheme="short_adjusted",
            notes="Short ranking penalises borrow tier and liquidity while the long side keeps raw score.",
        ),
        ExperimentSpec("D_short_leg", "baseline_tiered_borrow_cost", notes="Current tiered borrow cost."),
        ExperimentSpec("D_short_leg", "exclude_tier_c_shorts", short_treatment="exclude_tier_c"),
        ExperimentSpec("D_short_leg", "exclude_tier_b_and_c_shorts", short_treatment="exclude_tier_b_c"),
        ExperimentSpec("D_short_leg", "downweight_tier_b_c_shorts_50pct", short_treatment="downweight_b_c"),
        ExperimentSpec(
            "D_short_leg",
            "expand_short_basket_reduce_concentration",
            short_fraction=0.06,
            short_treatment="tiered_borrow",
        ),
        ExperimentSpec("E_target_transform", "raw_overnight_return", target_transform="raw"),
        ExperimentSpec(
            "E_target_transform",
            "cross_sectional_rank_target",
            target_transform="rank",
            notes="Current baseline target.",
        ),
        ExperimentSpec("E_target_transform", "demeaned_overnight_return", target_transform="demeaned"),
        ExperimentSpec("E_target_transform", "winsorised_overnight_return", target_transform="winsorised"),
        ExperimentSpec("E_target_transform", "volatility_scaled_overnight_return", target_transform="vol_scaled"),
    ]


def prepare_experiment_panel(config: StrategyConfig) -> tuple[pd.DataFrame, list[str]]:
    """Build the point-in-time panel shared by every experiment."""

    prepared = prepare_data(config)
    panel = add_point_in_time_features(prepared.panel)
    panel = apply_eligibility_filters(panel, config)
    panel, rank_columns = add_cross_sectional_ranks(panel)
    if panel["date"].max() > config.cutoff_ts:
        raise ValueError("Experiment panel includes data after the configured cutoff.")
    panel = slim_panel(panel, rank_columns)
    return panel, rank_columns


def slim_panel(panel: pd.DataFrame, rank_columns: list[str]) -> pd.DataFrame:
    """Keep only fields used by the experiment harness."""

    base_columns = [
        "date",
        "year",
        "instrument_id",
        "ticker",
        "is_trade_eligible",
        "overnight_next",
        "adv20",
        "vol20",
        "borrow_tier",
        "borrow_rate_annual",
    ]
    optional_columns = ["alpha_score", "predicted_alpha_return", TARGET_COLUMN]
    keep = [
        column
        for column in base_columns + rank_columns + optional_columns
        if column in panel.columns
    ]
    return panel.loc[:, keep].copy()


def cache_path(config: StrategyConfig, name: str) -> Path:
    safe_cutoff = config.cutoff_ts.strftime("%Y%m%d")
    return config.output_dir / f".strategy_experiment_{CACHE_VERSION}_{name}_{safe_cutoff}.parquet"


def load_or_prepare_experiment_panel(config: StrategyConfig) -> tuple[pd.DataFrame, list[str]]:
    """Load the shared experiment panel from cache or build it once."""

    path = cache_path(config, "panel")
    if path.exists():
        print(f"Loading cached shared panel from {path}...", flush=True)
        panel = pd.read_parquet(path)
        panel["date"] = pd.to_datetime(panel["date"])
        rank_columns = [column for column in panel.columns if column.startswith("rank_")]
        return slim_panel(panel, rank_columns), rank_columns

    panel, rank_columns = prepare_experiment_panel(config)
    print(f"Caching shared panel to {path}...", flush=True)
    panel.to_parquet(path, index=False)
    return panel, rank_columns


def add_target_transform(panel: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Create the training target for a controlled target-transform experiment."""

    panel = panel.copy()
    panel[TARGET_COLUMN] = np.nan
    mask = panel["is_trade_eligible"] & panel["overnight_next"].notna()
    eligible = panel.loc[mask, ["date", "overnight_next", "vol20"]].copy()

    if transform == "raw":
        target = eligible["overnight_next"]
    elif transform == "rank":
        target = eligible.groupby("date")["overnight_next"].rank(pct=True).sub(0.5)
    elif transform == "demeaned":
        target = eligible["overnight_next"] - eligible.groupby("date")["overnight_next"].transform(
            "mean"
        )
    elif transform == "winsorised":
        lower = eligible.groupby("date")["overnight_next"].transform(lambda x: x.quantile(0.01))
        upper = eligible.groupby("date")["overnight_next"].transform(lambda x: x.quantile(0.99))
        target = eligible["overnight_next"].clip(lower=lower, upper=upper)
    elif transform == "vol_scaled":
        daily_vol = (eligible["vol20"] / np.sqrt(252.0)).replace(0.0, np.nan)
        target = eligible["overnight_next"] / daily_vol
    else:
        raise ValueError(f"Unknown target transform: {transform}")

    panel.loc[mask, TARGET_COLUMN] = target.replace([np.inf, -np.inf], np.nan).astype(float)
    return panel


def learn_weights_for_target(
    train: pd.DataFrame, rank_columns: list[str], prior: pd.Series
) -> pd.Series:
    """Learn transparent feature weights against the selected target column."""

    mask = train["is_trade_eligible"] & train[TARGET_COLUMN].notna()
    sample = train.loc[mask, rank_columns + [TARGET_COLUMN]]
    if len(sample) < MIN_TRAINING_OBSERVATIONS:
        return prior

    matrix = sample[rank_columns].to_numpy(dtype=float)
    target = sample[TARGET_COLUMN].to_numpy(dtype=float)
    valid = np.isfinite(target)
    if valid.sum() < MIN_TRAINING_OBSERVATIONS:
        return prior

    matrix = np.nan_to_num(matrix[valid], nan=0.0, posinf=0.0, neginf=0.0)
    target = target[valid]
    target_centered = target - target.mean()
    target_ss = float(np.dot(target_centered, target_centered))
    if target_ss == 0:
        return prior

    matrix_centered = matrix - matrix.mean(axis=0)
    feature_ss = np.sum(matrix_centered * matrix_centered, axis=0)
    covariance = matrix_centered.T @ target_centered
    denominator = np.sqrt(feature_ss * target_ss)
    correlations = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 0,
    )

    learned = pd.Series(correlations, index=rank_columns).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    learned = (learned / 0.02).clip(-1.0, 1.0)
    weights = 0.50 * prior + 0.50 * learned.reindex(prior.index).fillna(0.0)
    total = weights.abs().sum()
    if total == 0:
        return prior
    return weights / total


def score_panel_with_target(
    panel: pd.DataFrame, rank_columns: list[str], config: StrategyConfig, transform: str
) -> pd.DataFrame:
    """Score the panel using the same walk-forward protocol with a different target."""

    scored = add_target_transform(panel, transform)
    scored["alpha_score"] = np.nan
    prior = _prior_weights(rank_columns)
    frozen_weights: pd.Series | None = None

    for year in sorted(scored["year"].dropna().unique()):
        year = int(year)
        print(f"  scoring {transform} target for year {year}...", flush=True)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = train_end - pd.DateOffset(years=config.training_years)

        if year_start > config.development_cutoff_ts and frozen_weights is not None:
            weights = frozen_weights
        else:
            train = scored.loc[scored["date"].between(train_start, train_end)]
            weights = learn_weights_for_target(train, rank_columns, prior)
            if year_start <= config.development_cutoff_ts:
                frozen_weights = weights

        mask = scored["year"].eq(year)
        matrix = scored.loc[mask, rank_columns].fillna(0.0).to_numpy(dtype=float)
        scored.loc[mask, "alpha_score"] = matrix @ weights.reindex(rank_columns).to_numpy()

    return scored


def load_or_score_target_panel(
    panel: pd.DataFrame, rank_columns: list[str], config: StrategyConfig, transform: str
) -> pd.DataFrame:
    """Load a scored target panel from cache or score and cache it."""

    path = cache_path(config, f"scored_{transform}")
    if path.exists():
        print(f"Loading cached scored panel for {transform} from {path}...", flush=True)
        scored = pd.read_parquet(path)
        scored["date"] = pd.to_datetime(scored["date"])
        return scored

    scored = score_panel_with_target(panel, rank_columns, config, transform)
    scored = add_walk_forward_alpha_return(scored, config)
    scored = slim_panel(scored.drop(columns=[TARGET_COLUMN], errors="ignore"), rank_columns)
    print(f"Caching scored panel for {transform} to {path}...", flush=True)
    scored.to_parquet(path, index=False)
    return scored


def add_walk_forward_alpha_return(panel: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Calibrate alpha scores into return units using past data only."""

    panel = panel.copy()
    panel["predicted_alpha_return"] = np.nan
    frozen_slope: float | None = None
    last_slope = DEFAULT_ALPHA_RETURN_SLOPE

    for year in sorted(panel["year"].dropna().unique()):
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = train_end - pd.DateOffset(years=config.training_years)

        if year_start > config.development_cutoff_ts and frozen_slope is not None:
            slope = frozen_slope
        else:
            train_mask = (
                panel["date"].between(train_start, train_end)
                & panel["is_trade_eligible"]
                & panel["alpha_score"].notna()
                & panel["overnight_next"].notna()
            )
            train = panel.loc[train_mask, ["alpha_score", "overnight_next"]]
            if len(train) >= MIN_TRAINING_OBSERVATIONS and train["alpha_score"].var() > 0:
                demeaned_target = train["overnight_next"] - train["overnight_next"].mean()
                slope = float(np.cov(train["alpha_score"], demeaned_target)[0, 1] / train["alpha_score"].var())
                if not np.isfinite(slope) or slope <= 0:
                    slope = last_slope
            else:
                slope = last_slope
            last_slope = slope
            if year_start <= config.development_cutoff_ts:
                frozen_slope = slope

        mask = panel["year"].eq(year)
        panel.loc[mask, "predicted_alpha_return"] = panel.loc[mask, "alpha_score"] * slope

    return panel


def capped_weighted_allocation(
    caps: np.ndarray, target_notional: float, preferences: np.ndarray | None = None
) -> np.ndarray:
    """Allocate notional by non-negative preferences subject to per-name caps."""

    caps = np.nan_to_num(caps.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    caps = np.maximum(caps, 0.0)
    if preferences is None:
        preferences = np.ones_like(caps)
    preferences = np.nan_to_num(preferences.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    preferences = np.maximum(preferences, 0.0)
    if preferences.sum() == 0:
        preferences = np.ones_like(caps)

    allocation = np.zeros_like(caps, dtype=float)
    remaining = np.arange(len(caps))
    target_left = min(float(target_notional), float(caps.sum()))

    while len(remaining) and target_left > 0:
        prefs = preferences[remaining]
        if prefs.sum() == 0:
            prefs = np.ones_like(prefs)
        desired = target_left * prefs / prefs.sum()
        capped = desired >= caps[remaining] - 1e-9
        if not capped.any():
            allocation[remaining] = desired
            break
        capped_indices = remaining[capped]
        allocation[capped_indices] = caps[capped_indices]
        target_left -= float(caps[capped_indices].sum())
        remaining = remaining[~capped]
    return allocation


def basket_size(count: int, fraction: float, config: StrategyConfig) -> int:
    """Convert a basket fraction into a non-overlapping count."""

    return max(config.min_basket_size, int(np.floor(count * fraction)))


def estimated_liquidity_cost(day: pd.DataFrame, aum: float, basket_count: int, config: StrategyConfig) -> pd.Series:
    """Estimate ex-ante daily-data trading cost for ranking only."""

    equal_notional = 0.5 * aum / max(basket_count, 1)
    participation = equal_notional / day["adv20"].replace(0.0, np.nan)
    participation = participation.clip(lower=0.0, upper=config.participation_cap).fillna(
        config.participation_cap
    )
    daily_vol = (day["vol20"].fillna(day["vol20"].median()) / np.sqrt(252.0)).clip(lower=0.0)
    impact_proxy = 0.7 * daily_vol * np.sqrt(participation)
    return (COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP + impact_proxy).astype(float)


def add_ranking_columns(day: pd.DataFrame, spec: ExperimentSpec, aum: float, config: StrategyConfig) -> pd.DataFrame:
    """Add long and short ranking scores for a single date."""

    day = day.copy()
    count = len(day)
    long_count = basket_size(count, spec.long_fraction, config)
    short_count = basket_size(count, spec.short_fraction, config)
    shared_count = max(long_count, short_count)
    liquidity_cost = estimated_liquidity_cost(day, aum, shared_count, config)

    if spec.ranking_scheme == "cost_adjusted":
        base = day["predicted_alpha_return"].fillna(day["alpha_score"])
        day["long_rank_score"] = base - liquidity_cost
        day["short_rank_score"] = base + liquidity_cost
        day["diagnostic_score"] = day["long_rank_score"]
    elif spec.ranking_scheme == "short_adjusted":
        borrow_penalty = day["borrow_rate_annual"].fillna(0.004) / 252.0
        day["long_rank_score"] = day["alpha_score"]
        day["short_rank_score"] = day["alpha_score"] + borrow_penalty + liquidity_cost
        day["diagnostic_score"] = day["alpha_score"]
    else:
        day["long_rank_score"] = day["alpha_score"]
        day["short_rank_score"] = day["alpha_score"]
        day["diagnostic_score"] = day["alpha_score"]
    return day


def choose_baskets(
    day: pd.DataFrame, spec: ExperimentSpec, aum: float, config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select long and short baskets under the experiment specification."""

    eligible = day.loc[
        day["is_trade_eligible"] & day["alpha_score"].notna() & day["overnight_next"].notna()
    ].copy()
    if len(eligible) < config.min_basket_size * 2:
        return eligible.iloc[0:0], eligible.iloc[0:0]

    eligible = add_ranking_columns(eligible, spec, aum, config)
    long_count = basket_size(len(eligible), spec.long_fraction, config)
    short_count = basket_size(len(eligible), spec.short_fraction, config)
    if long_count + short_count > len(eligible):
        scale = len(eligible) / float(long_count + short_count)
        long_count = max(config.min_basket_size, int(np.floor(long_count * scale)))
        short_count = max(config.min_basket_size, int(np.floor(short_count * scale)))

    longs = eligible.sort_values("long_rank_score", kind="mergesort").tail(long_count)
    short_pool = eligible.loc[~eligible["instrument_id"].isin(longs["instrument_id"])].copy()
    if spec.short_treatment == "exclude_tier_c":
        short_pool = short_pool.loc[short_pool["borrow_tier"].ne("C")]
    elif spec.short_treatment == "exclude_tier_b_c":
        short_pool = short_pool.loc[short_pool["borrow_tier"].eq("A")]

    if len(short_pool) < config.min_basket_size:
        return eligible.iloc[0:0], eligible.iloc[0:0]
    short_count = min(short_count, len(short_pool))
    shorts = short_pool.sort_values("short_rank_score", kind="mergesort").head(short_count)
    return longs, shorts


def score_preferences(frame: pd.DataFrame, side: str, scheme: str) -> np.ndarray:
    """Return within-basket sizing preferences."""

    n = len(frame)
    if n == 0:
        return np.array([])
    score = frame["alpha_score"].to_numpy(dtype=float)
    vol = frame["vol20"].replace(0.0, np.nan).fillna(frame["vol20"].median()).to_numpy(dtype=float)
    adv = frame["adv20"].replace(0.0, np.nan).fillna(frame["adv20"].median()).to_numpy(dtype=float)

    if side == "long":
        score_strength = score - np.nanmin(score)
    else:
        score_strength = np.nanmax(score) - score
    score_strength = np.nan_to_num(score_strength, nan=0.0)
    score_strength = np.maximum(score_strength, 0.0) + 1e-9

    if scheme == "volatility":
        prefs = 1.0 / np.maximum(vol, 1e-6)
    elif scheme == "score":
        prefs = score_strength
    elif scheme == "score_div_vol":
        prefs = score_strength / np.maximum(vol, 1e-6)
    elif scheme == "score_times_liquidity":
        prefs = score_strength * np.sqrt(np.maximum(adv, 1.0))
    else:
        prefs = np.ones(n)
    return np.nan_to_num(prefs, nan=0.0, posinf=0.0, neginf=0.0)


def size_baskets(
    longs: pd.DataFrame, shorts: pd.DataFrame, aum: float, config: StrategyConfig, spec: ExperimentSpec
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Size both baskets with dollar neutrality and the participation cap."""

    target_side = 0.5 * aum
    long_caps = (longs["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0.0)
    short_caps = (shorts["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0.0)
    if spec.short_treatment == "downweight_b_c":
        tier_multiplier = np.where(shorts["borrow_tier"].isin(["B", "C"]).to_numpy(), 0.5, 1.0)
        short_caps = short_caps * tier_multiplier

    long_preferences = score_preferences(longs, "long", spec.weighting_scheme)
    short_preferences = score_preferences(shorts, "short", spec.weighting_scheme)

    preliminary_long = capped_weighted_allocation(long_caps, target_side, long_preferences)
    preliminary_short = capped_weighted_allocation(short_caps, target_side, short_preferences)
    side_target = min(float(preliminary_long.sum()), float(preliminary_short.sum()), target_side)

    long_notional = capped_weighted_allocation(long_caps, side_target, long_preferences)
    short_notional = capped_weighted_allocation(short_caps, side_target, short_preferences)
    cap_binding = bool(
        np.any(long_notional >= long_caps - 1e-6)
        or np.any(short_notional >= short_caps - 1e-6)
        or side_target < target_side - 1e-6
    )
    return long_notional, short_notional, cap_binding


def run_experiment_backtest(
    panel: pd.DataFrame, aum: float, config: StrategyConfig, spec: ExperimentSpec
) -> pd.DataFrame:
    """Run one experiment and AUM through the same overnight cost schedule."""

    needed = [
        "date",
        "instrument_id",
        "ticker",
        "alpha_score",
        "predicted_alpha_return",
        "overnight_next",
        "adv20",
        "vol20",
        "borrow_tier",
        "borrow_rate_annual",
        "is_trade_eligible",
    ]
    work = panel[needed].sort_values("date", kind="mergesort")
    daily_rows: list[dict[str, float | pd.Timestamp | int | bool]] = []

    for date, day in work.groupby("date", sort=True):
        longs, shorts = choose_baskets(day, spec, aum, config)
        if longs.empty or shorts.empty:
            daily_rows.append(
                {
                    "date": date,
                    "gross_return": 0.0,
                    "commission_cost": 0.0,
                    "slippage_cost": 0.0,
                    "borrow_cost": 0.0,
                    "net_return": 0.0,
                    "turnover": 0.0,
                    "gross_exposure": 0.0,
                    "cap_binding": False,
                    "n_long": 0,
                    "n_short": 0,
                }
            )
            continue

        long_notional, short_notional, cap_binding = size_baskets(longs, shorts, aum, config, spec)
        long_weights = long_notional / aum
        short_weights = -short_notional / aum
        gross_exposure = float(long_weights.sum() + np.abs(short_weights).sum())

        gross_return = float(
            np.dot(long_weights, longs["overnight_next"].to_numpy(dtype=float))
            + np.dot(short_weights, shorts["overnight_next"].to_numpy(dtype=float))
        )
        commission_cost = gross_exposure * COMMISSION_ROUND_TRIP
        slippage_cost = gross_exposure * SLIPPAGE_ROUND_TRIP
        borrow_rates = shorts["borrow_rate_annual"].fillna(0.004).to_numpy(dtype=float)
        borrow_cost = float(np.dot(np.abs(short_weights), borrow_rates / 252.0))
        net_return = gross_return - commission_cost - slippage_cost - borrow_cost

        daily_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "commission_cost": commission_cost,
                "slippage_cost": slippage_cost,
                "borrow_cost": borrow_cost,
                "net_return": net_return,
                "turnover": 2.0 * gross_exposure,
                "gross_exposure": gross_exposure,
                "cap_binding": cap_binding,
                "n_long": int(len(longs)),
                "n_short": int(len(shorts)),
            }
        )

    return pd.DataFrame(daily_rows).sort_values("date")


def worst_rolling_12m_return(daily: pd.DataFrame) -> float:
    """Return the worst 252-session compounded net return."""

    returns = daily["net_return"].fillna(0.0)
    if len(returns) < 252:
        return np.nan
    rolling = (1.0 + returns).rolling(252).apply(np.prod, raw=True) - 1.0
    return float(rolling.min())


def ic_summary(panel: pd.DataFrame, score_column: str = "alpha_score") -> tuple[float, float]:
    """Compute daily Spearman IC mean and t-stat for a score column."""

    mask = (
        panel["is_trade_eligible"]
        & panel[score_column].notna()
        & panel["overnight_next"].notna()
    )
    work = panel.loc[mask, ["date", score_column, "overnight_next"]].copy()
    if work.empty:
        return np.nan, np.nan

    work["score_rank"] = work.groupby("date", sort=False)[score_column].rank(pct=True)
    work["return_rank"] = work.groupby("date", sort=False)["overnight_next"].rank(pct=True)
    work["score_rank_sq"] = work["score_rank"] * work["score_rank"]
    work["return_rank_sq"] = work["return_rank"] * work["return_rank"]
    work["rank_cross"] = work["score_rank"] * work["return_rank"]
    grouped = work.groupby("date", sort=False)
    stats = grouped.agg(
        n=("score_rank", "count"),
        sum_x=("score_rank", "sum"),
        sum_y=("return_rank", "sum"),
        sum_xx=("score_rank_sq", "sum"),
        sum_yy=("return_rank_sq", "sum"),
        sum_xy=("rank_cross", "sum"),
    )
    stats = stats.loc[stats["n"].ge(25)]
    if stats.empty:
        return np.nan, np.nan

    n = stats["n"].astype(float)
    cov = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    var_x = stats["sum_xx"] - stats["sum_x"] * stats["sum_x"] / n
    var_y = stats["sum_yy"] - stats["sum_y"] * stats["sum_y"] / n
    denom = np.sqrt(var_x * var_y)
    daily_ic = (cov / denom.replace(0.0, np.nan)).dropna()
    if daily_ic.empty or daily_ic.std() == 0:
        return np.nan, np.nan
    return float(daily_ic.mean()), float(daily_ic.mean() / daily_ic.std() * np.sqrt(len(daily_ic)))


def diagnostic_ic_panel(panel: pd.DataFrame, spec: ExperimentSpec, aum: float, config: StrategyConfig) -> pd.DataFrame:
    """Build a panel-level diagnostic score for IC reporting."""

    if spec.ranking_scheme != "cost_adjusted":
        return panel

    out = panel.copy()
    eligible_count = out.groupby("date")["is_trade_eligible"].transform("sum").clip(lower=1)
    basket_counts = np.maximum(config.min_basket_size, np.floor(eligible_count * spec.long_fraction))
    equal_notional = 0.5 * aum / basket_counts
    participation = (equal_notional / out["adv20"].replace(0.0, np.nan)).clip(
        lower=0.0, upper=config.participation_cap
    )
    daily_vol = (out["vol20"] / np.sqrt(252.0)).clip(lower=0.0)
    impact_proxy = 0.7 * daily_vol * np.sqrt(participation)
    cost = COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP + impact_proxy.fillna(0.0)
    out["diagnostic_score"] = out["predicted_alpha_return"].fillna(out["alpha_score"]) - cost
    return out.rename(columns={"alpha_score": "raw_alpha_score", "diagnostic_score": "alpha_score"})


def summarise_experiment(
    daily: pd.DataFrame,
    spec: ExperimentSpec,
    aum: float,
    ic_values: tuple[float, float],
) -> dict[str, float | str]:
    """Create one output row with the requested columns."""

    gross = daily["gross_return"]
    after_commission = gross - daily["commission_cost"]
    after_slippage = after_commission - daily["slippage_cost"]
    net = daily["net_return"]
    gross_sharpe = sharpe_ratio(gross)
    after_commission_sharpe = sharpe_ratio(after_commission)
    after_slippage_sharpe = sharpe_ratio(after_slippage)
    net_sharpe = sharpe_ratio(net)
    ic_mean, ic_tstat = ic_values

    return {
        "experiment_group": spec.experiment_group,
        "experiment_name": spec.experiment_name,
        "AUM": aum,
        "net_annual_return": annualised_return(net),
        "net_vol": annualised_volatility(net),
        "net_sharpe": net_sharpe,
        "max_drawdown": max_drawdown(net),
        "gross_sharpe": gross_sharpe,
        "commission_drag": gross_sharpe - after_commission_sharpe,
        "slippage_drag": after_commission_sharpe - after_slippage_sharpe,
        "borrow_drag": after_slippage_sharpe - net_sharpe,
        "average_daily_turnover": float(daily["turnover"].mean()),
        "average_gross_exposure_used": float(daily["gross_exposure"].mean()),
        "cap_binding_days": int(daily["cap_binding"].sum()),
        "average_long_names": float(daily["n_long"].mean()),
        "average_short_names": float(daily["n_short"].mean()),
        "IC_mean": ic_mean,
        "IC_tstat": ic_tstat,
        "worst_12m_return": worst_rolling_12m_return(daily),
        "notes": spec.notes,
    }


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None._"
    return frame[columns].to_markdown(index=False)


def source_diagnosis(row: pd.Series, baseline: pd.Series) -> str:
    """Classify the likely source of a 250M change versus baseline."""

    sources: list[str] = []
    if row["IC_mean"] > baseline["IC_mean"] + 0.002:
        sources.append("stronger alpha")
    if row["average_daily_turnover"] < baseline["average_daily_turnover"] - 0.05:
        sources.append("lower turnover/cost")
    if row["borrow_drag"] < baseline["borrow_drag"] - 0.02:
        sources.append("lower borrow drag")
    if row["average_gross_exposure_used"] > baseline["average_gross_exposure_used"] + 0.03:
        sources.append("better capacity use")
    if row["net_vol"] < baseline["net_vol"] - 0.003:
        sources.append("lower volatility")
    return ", ".join(sources) if sources else "mixed or small mechanical change"


def write_audit(summary: pd.DataFrame, path: Path) -> None:
    """Write the report-ready strategy improvement audit."""

    rows_250 = summary.loc[summary["AUM"].eq(BASELINE_AUM)].copy()
    baseline = rows_250.loc[
        (rows_250["experiment_group"].eq("A_basket_size"))
        & (rows_250["experiment_name"].eq("top_bottom_3pct"))
    ].iloc[0]

    improved = rows_250.loc[rows_250["net_sharpe"].gt(baseline["net_sharpe"] + 1e-12)].copy()
    improved["improvement_source"] = improved.apply(source_diagnosis, axis=1, baseline=baseline)
    improved = improved.sort_values("net_sharpe", ascending=False)

    robust = rows_250.loc[
        rows_250["net_sharpe"].le(baseline["net_sharpe"] + 1e-12)
        & (
            rows_250["worst_12m_return"].gt(baseline["worst_12m_return"])
            | rows_250["max_drawdown"].gt(baseline["max_drawdown"])
            | rows_250["net_vol"].lt(baseline["net_vol"])
        )
    ].copy()
    robust = robust.sort_values(["worst_12m_return", "max_drawdown"], ascending=False)

    unstable = rows_250.loc[
        rows_250["net_sharpe"].lt(0.0)
        | rows_250["IC_tstat"].lt(2.0)
        | rows_250["worst_12m_return"].lt(baseline["worst_12m_return"] - 0.05)
    ].copy()
    unstable = unstable.sort_values("net_sharpe")

    recommendation_pool = improved.loc[
        improved["IC_tstat"].ge(max(2.0, baseline["IC_tstat"] * 0.75))
        & improved["worst_12m_return"].ge(baseline["worst_12m_return"] - 0.03)
    ]
    if recommendation_pool.empty:
        recommended = baseline
        recommendation_text = (
            "Keep the current top/bottom 3% equal-weight, rank-target strategy. "
            "No logged variant improved Sharpe with enough robustness to justify replacing it."
        )
    else:
        recommended = recommendation_pool.iloc[0]
        recommendation_text = (
            f"Use `{recommended['experiment_group']} / {recommended['experiment_name']}` as the "
            "candidate final strategy, subject to final report review and no further unlogged tuning."
        )

    lines = [
        "# Strategy Improvement Audit",
        "",
        "Generated from `outputs/strategy_experiment_summary.csv` using the 2010-2024 development window only.",
        "",
        "## Baseline",
        "",
        (
            "Baseline is the current top/bottom 3% equal-weight 250M strategy with tiered borrow costs "
            "and cross-sectional rank target."
        ),
        "",
        (
            f"- Net Sharpe: {format_float(baseline['net_sharpe'])}; net annual return: "
            f"{format_float(baseline['net_annual_return'] * 100, 2)}%; max drawdown: "
            f"{format_float(baseline['max_drawdown'] * 100, 2)}%; worst 12m return: "
            f"{format_float(baseline['worst_12m_return'] * 100, 2)}%."
        ),
        "",
        "## 1. Changes That Improve 250M Net Sharpe",
        "",
        markdown_table(
            improved.head(12),
            [
                "experiment_group",
                "experiment_name",
                "net_sharpe",
                "net_annual_return",
                "net_vol",
                "max_drawdown",
                "worst_12m_return",
                "IC_mean",
                "improvement_source",
            ],
        ),
        "",
        "## 2. Robustness Improvements Without Higher Sharpe",
        "",
        markdown_table(
            robust.head(12),
            [
                "experiment_group",
                "experiment_name",
                "net_sharpe",
                "net_vol",
                "max_drawdown",
                "worst_12m_return",
                "average_gross_exposure_used",
                "average_daily_turnover",
            ],
        ),
        "",
        "## 3. Variants That Look Overfit Or Unstable",
        "",
        (
            "Flagged when 250M net Sharpe is negative, IC t-stat is below 2, or the worst rolling "
            "12-month return is more than 5 percentage points worse than baseline."
        ),
        "",
        markdown_table(
            unstable.head(12),
            [
                "experiment_group",
                "experiment_name",
                "net_sharpe",
                "IC_tstat",
                "max_drawdown",
                "worst_12m_return",
                "average_gross_exposure_used",
            ],
        ),
        "",
        "## 4. Source Of Improvement",
        "",
        (
            "The source label compares each 250M variant with baseline. It is a diagnostic rather than "
            "a causal proof: stronger alpha is proxied by higher IC, lower cost by lower turnover or drag, "
            "better capacity by higher gross exposure, and lower risk by lower realised volatility."
        ),
        "",
        markdown_table(
            improved.head(10),
            [
                "experiment_group",
                "experiment_name",
                "net_sharpe",
                "gross_sharpe",
                "commission_drag",
                "slippage_drag",
                "borrow_drag",
                "average_gross_exposure_used",
                "improvement_source",
            ],
        ),
        "",
        "## 5. Recommended Final Strategy Version",
        "",
        recommendation_text,
        "",
        (
            f"Recommended row: `{recommended['experiment_group']} / {recommended['experiment_name']}`; "
            f"250M net Sharpe {format_float(recommended['net_sharpe'])}, net annual return "
            f"{format_float(recommended['net_annual_return'] * 100, 2)}%, max drawdown "
            f"{format_float(recommended['max_drawdown'] * 100, 2)}%, worst 12m "
            f"{format_float(recommended['worst_12m_return'] * 100, 2)}%."
        ),
        "",
        "## Guardrails",
        "",
        "- No 2025 or later data is loaded; the configured cutoff is 2024-12-31.",
        "- All variants are logged in the CSV, including failed experiments.",
        "- No new features are introduced; all ranking adjustments use existing point-in-time score, volatility, ADV20, and borrow-tier fields observable at the decision date.",
        "- Headline returns still use the fixed Section 6.3 commission, slippage, and borrow schedule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        cutoff=args.cutoff,
        development_cutoff=args.development_cutoff,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "report_ready").mkdir(parents=True, exist_ok=True)

    print("Preparing shared point-in-time panel...", flush=True)
    panel, rank_columns = load_or_prepare_experiment_panel(config)

    print("Scoring baseline rank-target panel...", flush=True)
    baseline_scored = load_or_score_target_panel(panel, rank_columns, config, "rank")
    print("Computing baseline IC diagnostics...", flush=True)

    scored_by_target: dict[str, pd.DataFrame] = {"rank": baseline_scored}
    ic_by_target: dict[str, tuple[float, float]] = {"rank": ic_summary(baseline_scored)}
    print("Baseline panel ready; starting experiment grid...", flush=True)
    rows: list[dict[str, float | str]] = []
    output_path = config.output_dir / "strategy_experiment_summary.csv"

    for spec in experiment_specs():
        print(f"Running {spec.experiment_group} / {spec.experiment_name}...", flush=True)
        if spec.target_transform not in scored_by_target:
            print(f"Scoring target transform {spec.target_transform}...", flush=True)
            scored = load_or_score_target_panel(panel, rank_columns, config, spec.target_transform)
            scored_by_target[spec.target_transform] = scored
            print(f"Computing IC for target transform {spec.target_transform}...", flush=True)
            ic_by_target[spec.target_transform] = ic_summary(scored)
        scored_panel = scored_by_target[spec.target_transform]
        ic_values = ic_by_target[spec.target_transform]

        for aum in config.aums:
            daily = run_experiment_backtest(scored_panel, aum, config, spec)
            rows.append(summarise_experiment(daily, spec, aum, ic_values))
        pd.DataFrame(rows).to_csv(output_path, index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    audit_path = config.output_dir / "report_ready" / "strategy_improvement_audit.md"
    write_audit(summary, audit_path)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {audit_path}", flush=True)


if __name__ == "__main__":
    main()

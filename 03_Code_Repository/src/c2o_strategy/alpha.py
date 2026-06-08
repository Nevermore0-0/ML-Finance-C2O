"""Walk-forward transparent alpha model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig
from c2o_strategy.features import FEATURE_PRIORS


def add_target_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank realised next overnight returns within the eligible daily cross-section."""

    panel = panel.copy()
    mask = panel["is_trade_eligible"] & panel["overnight_next"].notna()
    panel["target_rank"] = np.nan
    ranks = (
        panel.loc[mask]
        .groupby("date")["overnight_next"]
        .rank(pct=True)
        .sub(0.5)
    )
    panel.loc[mask, "target_rank"] = ranks.astype(float)
    return panel


def _prior_weights(rank_columns: list[str]) -> pd.Series:
    priors = pd.Series({column: FEATURE_PRIORS.get(column, 0.0) for column in rank_columns})
    total = priors.abs().sum()
    if total == 0:
        return pd.Series(1.0 / len(rank_columns), index=rank_columns)
    return priors / total


def _learn_weights(
    train: pd.DataFrame, rank_columns: list[str], prior: pd.Series
) -> pd.Series:
    """Learn feature weights from past ranked features and ranked target."""

    mask = train["is_trade_eligible"] & train["target_rank"].notna()
    sample = train.loc[mask, rank_columns + ["target_rank"]]
    if len(sample) < 20_000:
        return prior

    correlations: dict[str, float] = {}
    target = sample["target_rank"].to_numpy(dtype=float)
    target_std = np.nanstd(target)
    for column in rank_columns:
        feature = sample[column].to_numpy(dtype=float)
        valid = np.isfinite(feature) & np.isfinite(target)
        if valid.sum() < 20_000 or np.nanstd(feature[valid]) == 0 or target_std == 0:
            correlations[column] = 0.0
            continue
        correlations[column] = float(np.corrcoef(feature[valid], target[valid])[0, 1])

    learned = pd.Series(correlations).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    learned = (learned / 0.02).clip(-1.0, 1.0)
    weights = 0.50 * prior + 0.50 * learned.reindex(prior.index).fillna(0.0)
    total = weights.abs().sum()
    if total == 0:
        return prior
    return weights / total


def score_panel(
    panel: pd.DataFrame, rank_columns: list[str], config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create scores using only data prior to the year being scored.

    For years after the development cutoff, weights are frozen at the final
    development-window estimate. This allows held-out 2025/2026 evaluation
    without contaminating model calibration.
    """

    panel = add_target_rank(panel)
    panel = panel.copy()
    panel["alpha_score"] = np.nan
    prior = _prior_weights(rank_columns)
    rows: list[dict[str, float | int | str]] = []
    frozen_weights: pd.Series | None = None

    years = sorted(panel["year"].dropna().unique())
    for year in years:
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = train_end - pd.DateOffset(years=config.training_years)

        if year_start > config.development_cutoff_ts and frozen_weights is not None:
            weights = frozen_weights
            source = "frozen_after_development_cutoff"
        else:
            train = panel.loc[panel["date"].between(train_start, train_end)]
            weights = _learn_weights(train, rank_columns, prior)
            source = "walk_forward_past_data" if len(train) else "prior"
            if year_start <= config.development_cutoff_ts:
                frozen_weights = weights

        mask = panel["year"].eq(year)
        matrix = panel.loc[mask, rank_columns].fillna(0.0).to_numpy(dtype=float)
        panel.loc[mask, "alpha_score"] = matrix @ weights.reindex(rank_columns).to_numpy()

        for column, weight in weights.items():
            rows.append(
                {
                    "year": year,
                    "feature": column,
                    "weight": float(weight),
                    "source": source,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                }
            )

    weights_frame = pd.DataFrame(rows)
    return panel, weights_frame


def compute_information_coefficients(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute daily Spearman IC between score and next overnight return."""

    mask = (
        panel["is_trade_eligible"]
        & panel["alpha_score"].notna()
        & panel["overnight_next"].notna()
    )
    rows: list[dict[str, float | pd.Timestamp | int]] = []
    for date, chunk in panel.loc[mask, ["date", "alpha_score", "overnight_next"]].groupby("date"):
        if len(chunk) < 25:
            continue
        ic = chunk["alpha_score"].corr(chunk["overnight_next"], method="spearman")
        rows.append({"date": date, "ic": float(ic), "n": int(len(chunk))})
    return pd.DataFrame(rows)


def summarise_ic(ic_daily: pd.DataFrame) -> pd.DataFrame:
    """Summarise IC by year."""

    if ic_daily.empty:
        return pd.DataFrame(columns=["year", "mean_ic", "ic_tstat", "days", "median_n"])
    out = ic_daily.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return (
        out.groupby("year")
        .agg(
            mean_ic=("ic", "mean"),
            ic_std=("ic", "std"),
            days=("ic", "count"),
            median_n=("n", "median"),
        )
        .reset_index()
        .assign(ic_tstat=lambda x: x["mean_ic"] / x["ic_std"] * np.sqrt(x["days"]))
        .drop(columns=["ic_std"])
    )


def summarise_ic_by_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute IC by macro-regime label where available."""

    if "regime" not in panel.columns:
        return pd.DataFrame()
    mask = (
        panel["is_trade_eligible"]
        & panel["alpha_score"].notna()
        & panel["overnight_next"].notna()
        & panel["regime"].notna()
    )
    rows: list[dict[str, float | str | int]] = []
    for regime, chunk in panel.loc[mask, ["date", "alpha_score", "overnight_next", "regime"]].groupby(
        "regime"
    ):
        daily = []
        for _, day in chunk.groupby("date"):
            if len(day) >= 25:
                daily.append(day["alpha_score"].corr(day["overnight_next"], method="spearman"))
        values = pd.Series(daily).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "regime": regime,
                "mean_ic": float(values.mean()),
                "ic_tstat": float(values.mean() / values.std() * np.sqrt(len(values)))
                if values.std() > 0
                else np.nan,
                "days": int(len(values)),
            }
        )
    return pd.DataFrame(rows)

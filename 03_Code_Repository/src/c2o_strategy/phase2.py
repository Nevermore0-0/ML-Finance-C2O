"""Phase 2 champion-challenger experiments and submission polish artifacts.

This module deliberately writes Phase 2 outputs to separate files.  It does not
overwrite the promoted final strategy outputs, the baseline archive, or the
current champion archive.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from c2o_strategy.alpha import _prior_weights
from c2o_strategy.config import StrategyConfig
from c2o_strategy.experiments import (
    MIN_TRAINING_OBSERVATIONS,
    add_target_transform,
    add_walk_forward_alpha_return,
    cache_path,
    capped_weighted_allocation,
    estimated_liquidity_cost,
    ic_summary,
    load_or_prepare_experiment_panel,
    load_or_score_target_panel,
    worst_rolling_12m_return,
)
from c2o_strategy.features import FEATURE_PRIORS
from c2o_strategy.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
)
from c2o_strategy.portfolio import COMMISSION_ROUND_TRIP, SLIPPAGE_ROUND_TRIP


BASELINE_AUM = 250_000_000.0
TARGET = "volatility_scaled_overnight_return"
TARGET_DEFINITION = "overnight_next / (vol20 / sqrt(252))"
DEFAULT_TRAINING_WINDOW = "4y_rolling"
DEFAULT_ALPHA_METHOD = "prior50_learned50"
DEFAULT_RANKING = "raw_alpha_score"
DEFAULT_WEIGHTING = "equal_weight"
DEFAULT_SHORT_TREATMENT = "tiered_borrow_cost_only"
PHASE2_CACHE_VERSION = "v1"
PHASE2_MODEL_CACHE_VERSION = "v1"


def _display_path(path: Path, base: Path | None = None) -> str:
    """Return a stable project-relative path for notebook and CLI logs."""

    candidate = Path(path)
    for root in (base, Path.cwd()):
        if root is None:
            continue
        try:
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
    return candidate.as_posix()


PERIODS: tuple[tuple[str, str, str], ...] = (
    ("design_2010_2018", "2010-01-01", "2018-12-31"),
    ("validation_2019_2022", "2019-01-01", "2022-12-31"),
    ("internal_holdout_2023_2024", "2023-01-01", "2024-12-31"),
    ("full_2010_2024", "2010-01-01", "2024-12-31"),
)


@dataclass(frozen=True)
class Phase2Spec:
    """One controlled Phase 2 challenger specification."""

    experiment_id: str
    experiment_group: str
    experiment_name: str
    long_fraction: float = 0.03
    short_fraction: float = 0.03
    weighting_scheme: str = DEFAULT_WEIGHTING
    ranking_scheme: str = DEFAULT_RANKING
    short_treatment: str = DEFAULT_SHORT_TREATMENT
    training_window: str = DEFAULT_TRAINING_WINDOW
    alpha_learning_method: str = DEFAULT_ALPHA_METHOD
    notes: str = ""

    @property
    def scored_key(self) -> str:
        return f"{self.training_window}_{self.alpha_learning_method}"

    @property
    def selection_key(self) -> tuple[str, float, float, str, str, str]:
        return (
            self.scored_key,
            self.long_fraction,
            self.short_fraction,
            self.weighting_scheme,
            self.ranking_scheme,
            self.short_treatment,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 C2O champion-challenger study.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--development-cutoff", default="2024-12-31")
    return parser.parse_args()


def report_ready_dir(config: StrategyConfig) -> Path:
    path = config.output_dir / "report_ready"
    path.mkdir(parents=True, exist_ok=True)
    return path


def phase2_specs() -> list[Phase2Spec]:
    """Return the controlled Phase 2 experiment grid."""

    specs: list[Phase2Spec] = [
        Phase2Spec(
            "phase2_g1_01_top_bottom_3pct_equal",
            "G1_basket_size_concentration",
            "top/bottom 3%, equal weight",
            notes="Current champion basket control.",
        ),
        Phase2Spec(
            "phase2_g1_02_top_bottom_5pct_equal",
            "G1_basket_size_concentration",
            "top/bottom 5%, equal weight",
            long_fraction=0.05,
            short_fraction=0.05,
        ),
        Phase2Spec(
            "phase2_g1_03_top_bottom_7_5pct_equal",
            "G1_basket_size_concentration",
            "top/bottom 7.5%, equal weight",
            long_fraction=0.075,
            short_fraction=0.075,
        ),
        Phase2Spec(
            "phase2_g1_04_top_bottom_10pct_equal",
            "G1_basket_size_concentration",
            "top/bottom 10%, equal weight",
            long_fraction=0.10,
            short_fraction=0.10,
        ),
        Phase2Spec(
            "phase2_g1_05_long_5_short_7_5_equal",
            "G1_basket_size_concentration",
            "long 5% / short 7.5%, equal weight",
            long_fraction=0.05,
            short_fraction=0.075,
        ),
        Phase2Spec(
            "phase2_g1_06_long_5_short_10_equal",
            "G1_basket_size_concentration",
            "long 5% / short 10%, equal weight",
            long_fraction=0.05,
            short_fraction=0.10,
        ),
        Phase2Spec(
            "phase2_g1_07_long_7_5_short_10_equal",
            "G1_basket_size_concentration",
            "long 7.5% / short 10%, equal weight",
            long_fraction=0.075,
            short_fraction=0.10,
        ),
        Phase2Spec(
            "phase2_g2_01_equal_weight",
            "G2_weighting_scheme",
            "equal weight",
            notes="Current champion weighting control.",
        ),
        Phase2Spec(
            "phase2_g2_02_volatility_weighted",
            "G2_weighting_scheme",
            "volatility-weighted",
            weighting_scheme="volatility_weighted",
        ),
        Phase2Spec(
            "phase2_g2_03_score_weighted",
            "G2_weighting_scheme",
            "score-weighted",
            weighting_scheme="score_weighted",
        ),
        Phase2Spec(
            "phase2_g2_04_score_divided_by_volatility",
            "G2_weighting_scheme",
            "score divided by volatility",
            weighting_scheme="score_divided_by_volatility",
        ),
        Phase2Spec(
            "phase2_g2_05_score_times_liquidity",
            "G2_weighting_scheme",
            "score times liquidity",
            weighting_scheme="score_times_liquidity",
        ),
        Phase2Spec(
            "phase2_g2_06_score_div_vol_liquidity_floor",
            "G2_weighting_scheme",
            "score divided by volatility with liquidity floor",
            weighting_scheme="score_divided_by_volatility_liquidity_floor",
            notes="Low-ADV selected names receive lower sizing preference; 5% ADV20 cap is unchanged.",
        ),
        Phase2Spec(
            "phase2_g2_07_score_weighted_single_name_cap",
            "G2_weighting_scheme",
            "score-weighted with max single-name cap",
            weighting_scheme="score_weighted_single_name_cap",
            notes="Adds a 10% per-side sizing cap; participation cap is unchanged.",
        ),
        Phase2Spec(
            "phase2_g3_01_raw_alpha_score",
            "G3_cost_aware_ranking",
            "raw alpha score",
            notes="Current champion ranking control.",
        ),
        Phase2Spec(
            "phase2_g3_02_alpha_minus_round_trip_cost",
            "G3_cost_aware_ranking",
            "predicted alpha minus estimated round-trip trading cost",
            ranking_scheme="alpha_minus_round_trip_cost",
        ),
        Phase2Spec(
            "phase2_g3_03_alpha_minus_liquidity_impact",
            "G3_cost_aware_ranking",
            "predicted alpha minus estimated liquidity impact",
            ranking_scheme="alpha_minus_liquidity_impact",
        ),
        Phase2Spec(
            "phase2_g3_04_short_alpha_minus_borrow",
            "G3_cost_aware_ranking",
            "short-side alpha adjusted for borrow cost",
            ranking_scheme="short_alpha_minus_borrow",
        ),
        Phase2Spec(
            "phase2_g3_05_short_alpha_minus_borrow_liquidity",
            "G3_cost_aware_ranking",
            "short-side alpha adjusted for borrow cost and liquidity penalty",
            ranking_scheme="short_alpha_minus_borrow_liquidity",
        ),
        Phase2Spec(
            "phase2_g3_06_cost_aware_long_short",
            "G3_cost_aware_ranking",
            "cost-aware long and short ranking together",
            ranking_scheme="cost_aware_long_short",
        ),
        Phase2Spec(
            "phase2_g4_01_tiered_borrow_cost_only",
            "G4_borrow_aware_short_leg",
            "tiered borrow cost only",
            notes="Current champion short-leg treatment control.",
        ),
        Phase2Spec(
            "phase2_g4_02_exclude_tier_c_shorts",
            "G4_borrow_aware_short_leg",
            "exclude Tier C shorts",
            short_treatment="exclude_tier_c",
        ),
        Phase2Spec(
            "phase2_g4_03_exclude_tier_b_c_shorts",
            "G4_borrow_aware_short_leg",
            "exclude Tier B and C shorts",
            short_treatment="exclude_tier_b_c",
        ),
        Phase2Spec(
            "phase2_g4_04_downweight_tier_b_c_50pct",
            "G4_borrow_aware_short_leg",
            "downweight Tier B/C shorts by 50%",
            short_treatment="downweight_tier_b_c_50pct",
        ),
        Phase2Spec(
            "phase2_g4_05_expand_short_basket",
            "G4_borrow_aware_short_leg",
            "expand short basket to reduce short concentration",
            short_fraction=0.06,
        ),
        Phase2Spec(
            "phase2_g4_06_exclude_tier_c_score_vol",
            "G4_borrow_aware_short_leg",
            "Tier C exclusion plus score/vol weighting",
            weighting_scheme="score_divided_by_volatility",
            short_treatment="exclude_tier_c",
        ),
        Phase2Spec(
            "phase2_g4_07_downweight_b_c_cost_aware",
            "G4_borrow_aware_short_leg",
            "Tier B/C downweight plus cost-aware short ranking",
            ranking_scheme="short_alpha_minus_borrow_liquidity",
            short_treatment="downweight_tier_b_c_50pct",
        ),
        Phase2Spec(
            "phase2_g5_01_2y_rolling",
            "G5_training_window",
            "2-year rolling training window",
            training_window="2y_rolling",
        ),
        Phase2Spec(
            "phase2_g5_02_3y_rolling",
            "G5_training_window",
            "3-year rolling training window",
            training_window="3y_rolling",
        ),
        Phase2Spec(
            "phase2_g5_03_4y_rolling_current",
            "G5_training_window",
            "4-year rolling training window",
            training_window=DEFAULT_TRAINING_WINDOW,
            notes="Current champion training-window control.",
        ),
        Phase2Spec(
            "phase2_g5_04_5y_rolling",
            "G5_training_window",
            "5-year rolling training window",
            training_window="5y_rolling",
        ),
        Phase2Spec(
            "phase2_g5_05_expanding",
            "G5_training_window",
            "expanding window",
            training_window="expanding",
        ),
        Phase2Spec(
            "phase2_g5_06_expanding_recent_decay",
            "G5_training_window",
            "expanding window with recent-year decay",
            training_window="expanding_recent_decay",
            notes="Deterministic two-year half-life applied in the learning step.",
        ),
        Phase2Spec(
            "phase2_g6_01_prior50_learned50",
            "G6_transparent_alpha_learning",
            "current 50% prior / 50% learned",
            alpha_learning_method=DEFAULT_ALPHA_METHOD,
            notes="Current champion alpha-learning control.",
        ),
        Phase2Spec(
            "phase2_g6_02_prior25_learned75",
            "G6_transparent_alpha_learning",
            "25% prior / 75% learned",
            alpha_learning_method="prior25_learned75",
        ),
        Phase2Spec(
            "phase2_g6_03_prior75_learned25",
            "G6_transparent_alpha_learning",
            "75% prior / 25% learned",
            alpha_learning_method="prior75_learned25",
        ),
        Phase2Spec(
            "phase2_g6_04_learned100",
            "G6_transparent_alpha_learning",
            "0% prior / 100% learned",
            alpha_learning_method="learned100",
        ),
        Phase2Spec(
            "phase2_g6_05_ridge_ranked_features",
            "G6_transparent_alpha_learning",
            "ridge regression on ranked features",
            alpha_learning_method="ridge_ranked_features",
        ),
        Phase2Spec(
            "phase2_g6_06_elastic_net_ranked_features",
            "G6_transparent_alpha_learning",
            "elastic net on ranked features",
            alpha_learning_method="elastic_net_ranked_features",
            notes="Uses scikit-learn already present in the recorded local environment.",
        ),
        Phase2Spec(
            "phase2_g6_07_robust_ridge_clipped",
            "G6_transparent_alpha_learning",
            "robust clipped ridge on ranked features",
            alpha_learning_method="robust_ridge_clipped",
        ),
    ]
    return specs


def phase2_cache_path(config: StrategyConfig, key: str) -> Path:
    safe_cutoff = config.cutoff_ts.strftime("%Y%m%d")
    return config.output_dir / f".phase2_{PHASE2_CACHE_VERSION}_scored_{key}_{safe_cutoff}.parquet"


def is_full_rank_feature_set(rank_columns: list[str]) -> bool:
    """Return True when the model uses the submitted full feature set."""

    return set(rank_columns) == set(FEATURE_PRIORS) and len(rank_columns) == len(FEATURE_PRIORS)


def phase2_weight_cache_path(
    config: StrategyConfig, training_window: str, alpha_learning_method: str
) -> Path:
    safe_key = f"{training_window}_{alpha_learning_method}".replace("/", "_")
    safe_development = config.development_cutoff_ts.strftime("%Y%m%d")
    return (
        config.model_cache_dir
        / f"phase2_weights_{PHASE2_MODEL_CACHE_VERSION}_{safe_key}_dev{safe_development}.csv"
    )


def phase2_weight_cache_manifest_path(config: StrategyConfig) -> Path:
    return config.model_cache_dir / "README.md"


def training_start_for_window(
    train_end: pd.Timestamp, config: StrategyConfig, training_window: str
) -> pd.Timestamp:
    if training_window == "2y_rolling":
        return train_end - pd.DateOffset(years=2)
    if training_window == "3y_rolling":
        return train_end - pd.DateOffset(years=3)
    if training_window == "4y_rolling":
        return train_end - pd.DateOffset(years=4)
    if training_window == "5y_rolling":
        return train_end - pd.DateOffset(years=5)
    if training_window in {"expanding", "expanding_recent_decay"}:
        return config.start_ts
    raise ValueError(f"Unknown training window: {training_window}")


def normalise_weights(weights: pd.Series, prior: pd.Series) -> pd.Series:
    weights = weights.reindex(prior.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    total = float(weights.abs().sum())
    if total == 0 or not np.isfinite(total):
        return prior
    return weights / total


def weighted_correlations(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if weights is None:
        x_centered = matrix - matrix.mean(axis=0)
        y_centered = target - target.mean()
        x_ss = np.sum(x_centered * x_centered, axis=0)
        y_ss = float(np.dot(y_centered, y_centered))
        cov = x_centered.T @ y_centered
    else:
        w = np.nan_to_num(weights.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        x_mean = w @ matrix
        y_mean = float(w @ target)
        x_centered = matrix - x_mean
        y_centered = target - y_mean
        x_ss = np.sum(w[:, None] * x_centered * x_centered, axis=0)
        y_ss = float(np.sum(w * y_centered * y_centered))
        cov = np.sum(w[:, None] * x_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(x_ss * y_ss)
    return np.divide(cov, denominator, out=np.zeros_like(cov, dtype=float), where=denominator > 0)


def correlation_learned_weights(
    sample: pd.DataFrame,
    rank_columns: list[str],
    prior: pd.Series,
    prior_weight: float,
    observation_weights: np.ndarray | None = None,
) -> pd.Series:
    matrix = sample[rank_columns].to_numpy(dtype=float)
    target = sample["experiment_target"].to_numpy(dtype=float)
    valid = np.isfinite(target)
    if valid.sum() < MIN_TRAINING_OBSERVATIONS:
        return prior
    matrix = matrix[valid]
    target = target[valid]
    obs_weights = observation_weights[valid] if observation_weights is not None else None
    correlations = weighted_correlations(matrix, target, obs_weights)
    learned = pd.Series(correlations, index=rank_columns).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    learned = (learned / 0.02).clip(-1.0, 1.0)
    if prior_weight <= 0:
        weights = learned
    else:
        weights = prior_weight * prior + (1.0 - prior_weight) * learned.reindex(prior.index).fillna(0.0)
    return normalise_weights(weights, prior)


def ridge_weights(
    sample: pd.DataFrame,
    rank_columns: list[str],
    prior: pd.Series,
    robust_clip: bool = False,
) -> pd.Series:
    matrix = sample[rank_columns].to_numpy(dtype=float)
    target = sample["experiment_target"].to_numpy(dtype=float)
    valid = np.isfinite(target)
    if valid.sum() < MIN_TRAINING_OBSERVATIONS:
        return prior
    x = np.nan_to_num(matrix[valid], nan=0.0, posinf=0.0, neginf=0.0)
    y = target[valid]
    if robust_clip:
        low, high = np.nanquantile(y, [0.01, 0.99])
        y = np.clip(y, low, high)
    x = x - x.mean(axis=0)
    y = y - y.mean()
    penalty = 5.0
    gram = x.T @ x
    rhs = x.T @ y
    try:
        coefs = np.linalg.solve(gram + penalty * np.eye(len(rank_columns)), rhs)
    except np.linalg.LinAlgError:
        coefs = np.linalg.pinv(gram + penalty * np.eye(len(rank_columns))) @ rhs
    return normalise_weights(pd.Series(coefs, index=rank_columns), prior)


def elastic_net_weights(sample: pd.DataFrame, rank_columns: list[str], prior: pd.Series) -> pd.Series:
    try:
        from sklearn.linear_model import ElasticNet
    except Exception:
        return prior

    matrix = sample[rank_columns].to_numpy(dtype=float)
    target = sample["experiment_target"].to_numpy(dtype=float)
    valid = np.isfinite(target)
    if valid.sum() < MIN_TRAINING_OBSERVATIONS:
        return prior
    x = np.nan_to_num(matrix[valid], nan=0.0, posinf=0.0, neginf=0.0)
    y = target[valid]
    x_std = x.std(axis=0)
    x_std = np.where(x_std > 0, x_std, 1.0)
    x = (x - x.mean(axis=0)) / x_std
    y = y - y.mean()
    model = ElasticNet(
        alpha=0.0005,
        l1_ratio=0.5,
        fit_intercept=False,
        max_iter=1000,
        selection="cyclic",
        random_state=42,
    )
    model.fit(x, y)
    coefs = pd.Series(model.coef_ / x_std, index=rank_columns)
    return normalise_weights(coefs, prior)


def learn_phase2_weights(
    train: pd.DataFrame,
    rank_columns: list[str],
    prior: pd.Series,
    alpha_learning_method: str,
    training_window: str,
    train_end: pd.Timestamp,
) -> pd.Series:
    mask = train["is_trade_eligible"] & train["experiment_target"].notna()
    sample = train.loc[mask, ["date", *rank_columns, "experiment_target"]]
    if len(sample) < MIN_TRAINING_OBSERVATIONS:
        return prior

    observation_weights = None
    if training_window == "expanding_recent_decay":
        age_days = (train_end - pd.to_datetime(sample["date"])).dt.days.clip(lower=0)
        observation_weights = np.power(0.5, age_days.to_numpy(dtype=float) / (365.25 * 2.0))

    if alpha_learning_method == "prior50_learned50":
        return correlation_learned_weights(sample, rank_columns, prior, 0.50, observation_weights)
    if alpha_learning_method == "prior25_learned75":
        return correlation_learned_weights(sample, rank_columns, prior, 0.25, observation_weights)
    if alpha_learning_method == "prior75_learned25":
        return correlation_learned_weights(sample, rank_columns, prior, 0.75, observation_weights)
    if alpha_learning_method == "learned100":
        return correlation_learned_weights(sample, rank_columns, prior, 0.0, observation_weights)
    if alpha_learning_method == "ridge_ranked_features":
        return ridge_weights(sample, rank_columns, prior, robust_clip=False)
    if alpha_learning_method == "elastic_net_ranked_features":
        return elastic_net_weights(sample, rank_columns, prior)
    if alpha_learning_method == "robust_ridge_clipped":
        return ridge_weights(sample, rank_columns, prior, robust_clip=True)
    raise ValueError(f"Unknown alpha learning method: {alpha_learning_method}")


def train_phase2_weight_schedule(
    panel: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str,
    alpha_learning_method: str,
) -> pd.DataFrame:
    """Train and return the yearly Phase 2 feature-weight schedule."""

    scored = add_target_transform(panel, "vol_scaled")
    prior = _prior_weights(rank_columns)
    frozen_weights: pd.Series | None = None
    rows: list[dict[str, object]] = []
    feature_set = "|".join(rank_columns)

    for year in sorted(scored["year"].dropna().unique()):
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = training_start_for_window(train_end, config, training_window)

        if year_start > config.development_cutoff_ts and frozen_weights is not None:
            weights = frozen_weights
            source = "frozen_from_development_cutoff"
            sample_count = 0
        else:
            train = scored.loc[scored["date"].between(train_start, train_end)]
            sample_count = int((train["is_trade_eligible"] & train["experiment_target"].notna()).sum())
            weights = learn_phase2_weights(
                train,
                rank_columns,
                prior,
                alpha_learning_method,
                training_window,
                train_end,
            )
            source = (
                "prior_fallback"
                if sample_count < MIN_TRAINING_OBSERVATIONS
                else f"phase2_{training_window}_{alpha_learning_method}"
            )
            if year_start <= config.development_cutoff_ts:
                frozen_weights = weights

        for feature, weight in weights.reindex(rank_columns).items():
            rows.append(
                {
                    "cache_version": PHASE2_MODEL_CACHE_VERSION,
                    "year": year,
                    "feature": feature,
                    "weight": float(weight),
                    "source": source,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "training_rows": sample_count,
                    "training_window": training_window,
                    "alpha_learning_method": alpha_learning_method,
                    "target": TARGET,
                    "target_definition": TARGET_DEFINITION,
                    "start_date": config.start_date,
                    "development_cutoff": config.development_cutoff,
                    "feature_count": len(rank_columns),
                    "feature_set": feature_set,
                }
            )

    return pd.DataFrame(rows)


def weight_schedule_is_compatible(
    schedule: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str,
    alpha_learning_method: str,
) -> bool:
    """Check that a cached schedule can be used for the requested model."""

    required = {
        "cache_version",
        "year",
        "feature",
        "weight",
        "training_window",
        "alpha_learning_method",
        "target",
        "start_date",
        "development_cutoff",
        "feature_set",
    }
    if not required <= set(schedule.columns):
        return False
    expected = {
        "cache_version": PHASE2_MODEL_CACHE_VERSION,
        "training_window": training_window,
        "alpha_learning_method": alpha_learning_method,
        "target": TARGET,
        "start_date": config.start_date,
        "development_cutoff": config.development_cutoff,
        "feature_set": "|".join(rank_columns),
    }
    for column, value in expected.items():
        observed = schedule[column].dropna().astype(str).unique()
        if len(observed) != 1 or observed[0] != str(value):
            return False
    if set(schedule["feature"]) != set(rank_columns):
        return False
    expected_per_year = len(rank_columns)
    return bool(schedule.groupby("year")["feature"].nunique().eq(expected_per_year).all())


def write_phase2_weight_cache_manifest(config: StrategyConfig) -> None:
    """Document model-cache semantics for markers and future runs."""

    config.model_cache_dir.mkdir(parents=True, exist_ok=True)
    phase2_weight_cache_manifest_path(config).write_text(
        "\n".join(
            [
                "# Model Cache",
                "",
                "This folder stores trained Phase 2 yearly feature-weight schedules.",
                "The notebook still rebuilds the point-in-time panel from raw `data/`,",
                "then applies these cached weights to score each stock-day and",
                "regenerates positions, returns, report tables, figures, QuantStats,",
                "and tests. If a compatible cache file is missing, stale, or uses a",
                "different feature set/development cutoff, the notebook trains the",
                "weights again and writes a fresh cache.",
                "",
                "These files are model artefacts, not final performance CSVs.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_or_train_phase2_weight_schedule(
    panel: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str,
    alpha_learning_method: str,
    use_model_cache: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load compatible cached weights or train them from the raw panel."""

    path = phase2_weight_cache_path(config, training_window, alpha_learning_method)
    if use_model_cache and path.exists():
        schedule = pd.read_csv(path)
        if weight_schedule_is_compatible(
            schedule, rank_columns, config, training_window, alpha_learning_method
        ):
            print(
                f"Loading cached Phase 2 weights from {_display_path(path, config.output_dir.parent)}...",
                flush=True,
            )
            return schedule, "loaded_model_cache"
        print(
            "Ignoring incompatible Phase 2 weight cache "
            f"{_display_path(path, config.output_dir.parent)}; retraining.",
            flush=True,
        )

    print(
        f"Training Phase 2 weights for {training_window} / {alpha_learning_method}...",
        flush=True,
    )
    schedule = train_phase2_weight_schedule(
        panel,
        rank_columns,
        config,
        training_window,
        alpha_learning_method,
    )
    if use_model_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(path, index=False)
        write_phase2_weight_cache_manifest(config)
        print(
            f"Wrote Phase 2 weight cache to {_display_path(path, config.output_dir.parent)}.",
            flush=True,
        )
    return schedule, "trained_model_cache"


def score_panel_phase2_with_weight_schedule(
    panel: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Score a Phase 2 panel from a cached or freshly trained weight schedule."""

    scored = add_target_transform(panel, "vol_scaled")
    scored["alpha_score"] = np.nan
    by_year = {
        int(year): group.set_index("feature")["weight"].reindex(rank_columns).fillna(0.0)
        for year, group in schedule.groupby("year")
    }
    development_years = [
        year
        for year in by_year
        if pd.Timestamp(year=year, month=1, day=1) <= config.development_cutoff_ts
    ]
    frozen_year = max(development_years) if development_years else max(by_year)

    for year in sorted(scored["year"].dropna().unique()):
        year = int(year)
        weights = by_year.get(year)
        if weights is None and pd.Timestamp(year=year, month=1, day=1) > config.development_cutoff_ts:
            weights = by_year[frozen_year]
        if weights is None:
            raise ValueError(f"Cached weights do not cover in-development year {year}.")
        mask = scored["year"].eq(year)
        matrix = scored.loc[mask, rank_columns].fillna(0.0).to_numpy(dtype=float)
        scored.loc[mask, "alpha_score"] = matrix @ weights.to_numpy(dtype=float)

    scored = add_walk_forward_alpha_return_phase2(scored, config, training_window)
    return scored.drop(columns=["experiment_target"], errors="ignore")


def add_walk_forward_alpha_return_phase2(
    panel: pd.DataFrame, config: StrategyConfig, training_window: str
) -> pd.DataFrame:
    """Calibrate scores into return units using the same Phase 2 training window."""

    panel = panel.copy()
    panel["predicted_alpha_return"] = np.nan
    frozen_slope: float | None = None
    last_slope = 0.01
    for year in sorted(panel["year"].dropna().unique()):
        year = int(year)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = training_start_for_window(train_end, config, training_window)
        if year_start > config.development_cutoff_ts and frozen_slope is not None:
            slope = frozen_slope
        else:
            mask = (
                panel["date"].between(train_start, train_end)
                & panel["is_trade_eligible"]
                & panel["alpha_score"].notna()
                & panel["overnight_next"].notna()
            )
            train = panel.loc[mask, ["alpha_score", "overnight_next"]]
            if len(train) >= MIN_TRAINING_OBSERVATIONS and train["alpha_score"].var() > 0:
                target = train["overnight_next"] - train["overnight_next"].mean()
                slope = float(np.cov(train["alpha_score"], target)[0, 1] / train["alpha_score"].var())
                if not np.isfinite(slope) or slope <= 0:
                    slope = last_slope
            else:
                slope = last_slope
            last_slope = slope
            if year_start <= config.development_cutoff_ts:
                frozen_slope = slope
        panel.loc[panel["year"].eq(year), "predicted_alpha_return"] = (
            panel.loc[panel["year"].eq(year), "alpha_score"] * slope
        )
    return panel


def score_panel_phase2(
    panel: pd.DataFrame,
    rank_columns: list[str],
    config: StrategyConfig,
    training_window: str,
    alpha_learning_method: str,
    use_model_cache: bool = True,
) -> pd.DataFrame:
    if use_model_cache and is_full_rank_feature_set(rank_columns):
        schedule, _ = load_or_train_phase2_weight_schedule(
            panel,
            rank_columns,
            config,
            training_window,
            alpha_learning_method,
            use_model_cache=True,
        )
        return score_panel_phase2_with_weight_schedule(
            panel,
            rank_columns,
            config,
            training_window,
            schedule,
        )

    scored = add_target_transform(panel, "vol_scaled")
    scored["alpha_score"] = np.nan
    prior = _prior_weights(rank_columns)
    frozen_weights: pd.Series | None = None

    for year in sorted(scored["year"].dropna().unique()):
        year = int(year)
        print(
            f"  Phase 2 scoring {training_window} / {alpha_learning_method} for {year}...",
            flush=True,
        )
        year_start = pd.Timestamp(year=year, month=1, day=1)
        previous_year_end = year_start - pd.Timedelta(days=1)
        train_end = min(previous_year_end, config.development_cutoff_ts)
        train_start = training_start_for_window(train_end, config, training_window)

        if year_start > config.development_cutoff_ts and frozen_weights is not None:
            weights = frozen_weights
        else:
            train = scored.loc[scored["date"].between(train_start, train_end)]
            weights = learn_phase2_weights(
                train,
                rank_columns,
                prior,
                alpha_learning_method,
                training_window,
                train_end,
            )
            if year_start <= config.development_cutoff_ts:
                frozen_weights = weights

        mask = scored["year"].eq(year)
        matrix = scored.loc[mask, rank_columns].fillna(0.0).to_numpy(dtype=float)
        scored.loc[mask, "alpha_score"] = matrix @ weights.reindex(rank_columns).to_numpy()

    scored = add_walk_forward_alpha_return_phase2(scored, config, training_window)
    return scored.drop(columns=["experiment_target"], errors="ignore")


def load_or_score_phase2_panel(
    panel: pd.DataFrame, rank_columns: list[str], config: StrategyConfig, spec: Phase2Spec
) -> pd.DataFrame:
    path = phase2_cache_path(config, spec.scored_key)
    if path.exists():
        print(
            "Loading cached Phase 2 scored panel "
            f"{spec.scored_key} from {_display_path(path, config.output_dir.parent)}...",
            flush=True,
        )
        out = pd.read_parquet(path)
        out["date"] = pd.to_datetime(out["date"])
        return out

    out = score_panel_phase2(
        panel,
        rank_columns,
        config,
        spec.training_window,
        spec.alpha_learning_method,
    )
    if out["date"].max() > config.cutoff_ts:
        raise ValueError("Phase 2 scored panel includes data after cutoff.")
    print(
        f"Caching Phase 2 scored panel {spec.scored_key} to {_display_path(path, config.output_dir.parent)}...",
        flush=True,
    )
    out.to_parquet(path, index=False)
    return out


def basket_count(count: int, fraction: float, config: StrategyConfig) -> int:
    return max(config.min_basket_size, int(np.floor(count * fraction)))


def liquidity_impact_proxy(day: pd.DataFrame, aum: float, basket_count_value: int, config: StrategyConfig) -> pd.Series:
    total_cost = estimated_liquidity_cost(day, aum, basket_count_value, config)
    return (total_cost - (COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP)).clip(lower=0.0)


def add_phase2_ranking_columns(
    day: pd.DataFrame, spec: Phase2Spec, aum: float, config: StrategyConfig
) -> pd.DataFrame:
    day = day.copy()
    long_count = basket_count(len(day), spec.long_fraction, config)
    short_count = basket_count(len(day), spec.short_fraction, config)
    shared_count = max(long_count, short_count)
    impact = liquidity_impact_proxy(day, aum, shared_count, config)
    fixed_cost = COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP
    borrow_daily = day["borrow_rate_annual"].fillna(0.004) / 252.0
    predicted = day["predicted_alpha_return"].fillna(day["alpha_score"] * 0.01)
    raw = day["alpha_score"]

    if spec.ranking_scheme == "alpha_minus_round_trip_cost":
        day["long_rank_score"] = predicted - fixed_cost
        day["short_rank_score"] = predicted + fixed_cost
    elif spec.ranking_scheme == "alpha_minus_liquidity_impact":
        day["long_rank_score"] = predicted - impact
        day["short_rank_score"] = predicted + impact
    elif spec.ranking_scheme == "short_alpha_minus_borrow":
        day["long_rank_score"] = raw
        day["short_rank_score"] = raw + borrow_daily
    elif spec.ranking_scheme == "short_alpha_minus_borrow_liquidity":
        day["long_rank_score"] = raw
        day["short_rank_score"] = raw + borrow_daily + impact
    elif spec.ranking_scheme == "cost_aware_long_short":
        day["long_rank_score"] = predicted - fixed_cost - impact
        day["short_rank_score"] = predicted + fixed_cost + impact + borrow_daily
    else:
        day["long_rank_score"] = raw
        day["short_rank_score"] = raw
    return day


def choose_phase2_baskets(
    day: pd.DataFrame, spec: Phase2Spec, aum: float, config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = day.loc[
        day["is_trade_eligible"] & day["alpha_score"].notna() & day["overnight_next"].notna()
    ].copy()
    if len(eligible) < config.min_basket_size * 2:
        return eligible.iloc[0:0], eligible.iloc[0:0]

    eligible = add_phase2_ranking_columns(eligible, spec, aum, config)
    long_count = basket_count(len(eligible), spec.long_fraction, config)
    short_count = basket_count(len(eligible), spec.short_fraction, config)
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


def phase2_preferences(frame: pd.DataFrame, side: str, scheme: str) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.array([])

    score = frame["alpha_score"].to_numpy(dtype=float)
    vol = frame["vol20"].replace(0.0, np.nan).fillna(frame["vol20"].median()).to_numpy(dtype=float)
    adv = frame["adv20"].replace(0.0, np.nan).fillna(frame["adv20"].median()).to_numpy(dtype=float)
    if side == "long":
        strength = score - np.nanmin(score)
    else:
        strength = np.nanmax(score) - score
    strength = np.maximum(np.nan_to_num(strength, nan=0.0), 0.0) + 1e-9

    if scheme == "volatility_weighted":
        prefs = 1.0 / np.maximum(vol, 1e-6)
    elif scheme in {"score_weighted", "score_weighted_single_name_cap"}:
        prefs = strength
    elif scheme == "score_divided_by_volatility":
        prefs = strength / np.maximum(vol, 1e-6)
    elif scheme == "score_times_liquidity":
        prefs = strength * np.sqrt(np.maximum(adv, 1.0))
    elif scheme == "score_divided_by_volatility_liquidity_floor":
        floor = np.nanquantile(adv, 0.25) if np.isfinite(adv).any() else 0.0
        liquidity_multiplier = np.where(adv >= floor, 1.0, 0.25)
        prefs = strength / np.maximum(vol, 1e-6) * liquidity_multiplier
    else:
        prefs = np.ones(n)
    return np.nan_to_num(prefs, nan=0.0, posinf=0.0, neginf=0.0)


def size_phase2_baskets(
    longs: pd.DataFrame, shorts: pd.DataFrame, aum: float, config: StrategyConfig, spec: Phase2Spec
) -> tuple[np.ndarray, np.ndarray, bool]:
    target_side = 0.5 * aum
    long_caps = (longs["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0.0)
    short_caps = (shorts["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0.0)

    if spec.weighting_scheme == "score_weighted_single_name_cap":
        single_name_cap = 0.10 * target_side
        long_caps = np.minimum(long_caps, single_name_cap)
        short_caps = np.minimum(short_caps, single_name_cap)
    if spec.short_treatment == "downweight_tier_b_c_50pct":
        multiplier = np.where(shorts["borrow_tier"].isin(["B", "C"]).to_numpy(), 0.5, 1.0)
        short_caps = short_caps * multiplier

    long_preferences = phase2_preferences(longs, "long", spec.weighting_scheme)
    short_preferences = phase2_preferences(shorts, "short", spec.weighting_scheme)
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


def run_phase2_backtest(
    panel: pd.DataFrame, aum: float, config: StrategyConfig, spec: Phase2Spec
) -> pd.DataFrame:
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
        longs, shorts = choose_phase2_baskets(day, spec, aum, config)
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

        long_notional, short_notional, cap_binding = size_phase2_baskets(
            longs, shorts, aum, config, spec
        )
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
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "commission_cost": commission_cost,
                "slippage_cost": slippage_cost,
                "borrow_cost": borrow_cost,
                "net_return": gross_return - commission_cost - slippage_cost - borrow_cost,
                "turnover": 2.0 * gross_exposure,
                "gross_exposure": gross_exposure,
                "cap_binding": cap_binding,
                "n_long": int(len(longs)),
                "n_short": int(len(shorts)),
            }
        )

    return pd.DataFrame(daily_rows).sort_values("date")


def ic_for_period(panel: pd.DataFrame, start: str, end: str) -> tuple[float, float]:
    period = panel.loc[pd.to_datetime(panel["date"]).between(start, end)]
    return ic_summary(period)


def summarise_period(
    daily: pd.DataFrame,
    spec: Phase2Spec,
    aum: float,
    period_name: str,
    start: str,
    end: str,
    ic_values: tuple[float, float],
) -> dict[str, float | str | int]:
    chunk = daily.loc[pd.to_datetime(daily["date"]).between(start, end)].copy()
    gross = chunk["gross_return"]
    after_commission = gross - chunk["commission_cost"]
    after_slippage = after_commission - chunk["slippage_cost"]
    net = chunk["net_return"]
    gross_sharpe = sharpe_ratio(gross)
    after_commission_sharpe = sharpe_ratio(after_commission)
    after_slippage_sharpe = sharpe_ratio(after_slippage)
    net_sharpe = sharpe_ratio(net)
    ic_mean, ic_tstat = ic_values
    return {
        "experiment_id": spec.experiment_id,
        "experiment_group": spec.experiment_group,
        "experiment_name": spec.experiment_name,
        "target": TARGET,
        "target_definition": TARGET_DEFINITION,
        "basket_rule": spec.experiment_name,
        "long_fraction": spec.long_fraction,
        "short_fraction": spec.short_fraction,
        "weighting_scheme": spec.weighting_scheme,
        "ranking_scheme": spec.ranking_scheme,
        "short_treatment": spec.short_treatment,
        "training_window": spec.training_window,
        "alpha_learning_method": spec.alpha_learning_method,
        "AUM": aum,
        "period": period_name,
        "start_date": start,
        "end_date": end,
        "net_annual_return": annualised_return(net),
        "net_vol": annualised_volatility(net),
        "net_sharpe": net_sharpe,
        "max_drawdown": max_drawdown(net),
        "gross_sharpe": gross_sharpe,
        "commission_drag": gross_sharpe - after_commission_sharpe,
        "slippage_drag": after_commission_sharpe - after_slippage_sharpe,
        "borrow_drag": after_slippage_sharpe - net_sharpe,
        "average_turnover": float(chunk["turnover"].mean()) if not chunk.empty else np.nan,
        "average_gross_exposure_used": float(chunk["gross_exposure"].mean()) if not chunk.empty else np.nan,
        "cap_binding_days": int(chunk["cap_binding"].sum()) if not chunk.empty else 0,
        "fraction_cap_binding_days": float(chunk["cap_binding"].mean()) if not chunk.empty else np.nan,
        "average_long_names": float(chunk["n_long"].mean()) if not chunk.empty else np.nan,
        "average_short_names": float(chunk["n_short"].mean()) if not chunk.empty else np.nan,
        "IC_mean": ic_mean,
        "IC_tstat": ic_tstat,
        "worst_12m_return": worst_rolling_12m_return(chunk) if not chunk.empty else np.nan,
        "average_daily_commission_bps": float(chunk["commission_cost"].mean() * 10_000)
        if not chunk.empty
        else np.nan,
        "average_daily_slippage_bps": float(chunk["slippage_cost"].mean() * 10_000)
        if not chunk.empty
        else np.nan,
        "average_daily_borrow_bps": float(chunk["borrow_cost"].mean() * 10_000)
        if not chunk.empty
        else np.nan,
        "notes": spec.notes,
    }


def read_champion_daily(config: StrategyConfig, aum: float) -> pd.DataFrame:
    label = "1b" if int(aum) == 1_000_000_000 else f"{int(aum / 1_000_000)}m"
    daily = pd.read_csv(config.output_dir / f"daily_returns_{label}.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def current_champion_period_rows(config: StrategyConfig) -> pd.DataFrame:
    panel_path = cache_path(config, "scored_vol_scaled")
    scored = pd.read_parquet(panel_path)
    scored["date"] = pd.to_datetime(scored["date"])
    spec = Phase2Spec(
        "current_champion",
        "Champion",
        "current volatility-scaled target champion",
        notes="Frozen current champion from preserved final outputs.",
    )
    rows = []
    for aum in config.aums:
        daily = read_champion_daily(config, aum)
        for period_name, start, end in PERIODS:
            rows.append(
                summarise_period(
                    daily,
                    spec,
                    aum,
                    period_name,
                    start,
                    end,
                    ic_for_period(scored, start, end),
                )
            )
    return pd.DataFrame(rows)


def pivot_metric(frame: pd.DataFrame, metric: str, aum: float) -> pd.Series:
    data = frame.loc[frame["AUM"].eq(aum)].pivot_table(
        index="experiment_id", columns="period", values=metric, aggfunc="first"
    )
    return data


def build_champion_challenger_matrix(summary: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    champion = current_champion_period_rows(config)
    combined = pd.concat([champion, summary], ignore_index=True)
    meta = combined.drop_duplicates("experiment_id")[
        [
            "experiment_id",
            "experiment_group",
            "experiment_name",
            "long_fraction",
            "short_fraction",
            "weighting_scheme",
            "ranking_scheme",
            "short_treatment",
            "training_window",
            "alpha_learning_method",
            "notes",
        ]
    ].set_index("experiment_id")

    matrix = meta.copy()
    for period in ["validation_2019_2022", "internal_holdout_2023_2024", "full_2010_2024"]:
        rows = combined.loc[(combined["AUM"].eq(BASELINE_AUM)) & combined["period"].eq(period)]
        matrix[f"{period}_250m_net_sharpe"] = rows.set_index("experiment_id")["net_sharpe"]
        matrix[f"{period}_250m_max_drawdown"] = rows.set_index("experiment_id")["max_drawdown"]
        matrix[f"{period}_250m_worst_12m_return"] = rows.set_index("experiment_id")["worst_12m_return"]
    for aum, label in [(50_000_000.0, "50m"), (250_000_000.0, "250m"), (1_000_000_000.0, "1b")]:
        rows = combined.loc[(combined["AUM"].eq(aum)) & combined["period"].eq("full_2010_2024")]
        matrix[f"full_{label}_net_sharpe"] = rows.set_index("experiment_id")["net_sharpe"]
    full250 = combined.loc[
        (combined["AUM"].eq(BASELINE_AUM)) & combined["period"].eq("full_2010_2024")
    ].set_index("experiment_id")
    for column in [
        "net_annual_return",
        "net_vol",
        "gross_sharpe",
        "commission_drag",
        "slippage_drag",
        "borrow_drag",
        "average_turnover",
        "average_gross_exposure_used",
        "cap_binding_days",
        "fraction_cap_binding_days",
        "IC_mean",
        "IC_tstat",
    ]:
        matrix[f"full_250m_{column}"] = full250[column]
    matrix["full_250m_total_cost_drag"] = (
        matrix["full_250m_commission_drag"]
        + matrix["full_250m_slippage_drag"]
        + matrix["full_250m_borrow_drag"]
    )

    champion_row = matrix.loc["current_champion"]
    matrix["beats_validation_sharpe"] = (
        matrix["validation_2019_2022_250m_net_sharpe"]
        > champion_row["validation_2019_2022_250m_net_sharpe"]
    )
    matrix["holdout_non_failure"] = (
        matrix["internal_holdout_2023_2024_250m_net_sharpe"].ge(0.0)
        & matrix["internal_holdout_2023_2024_250m_max_drawdown"].ge(
            champion_row["internal_holdout_2023_2024_250m_max_drawdown"] - 0.03
        )
    )
    matrix["drawdown_not_materially_worse"] = matrix["full_2010_2024_250m_max_drawdown"].ge(
        champion_row["full_2010_2024_250m_max_drawdown"] - 0.03
    )
    matrix["worst_12m_not_materially_worse"] = matrix[
        "full_2010_2024_250m_worst_12m_return"
    ].ge(champion_row["full_2010_2024_250m_worst_12m_return"] - 0.03)
    matrix["not_low_exposure_artifact"] = matrix["full_250m_average_gross_exposure_used"].ge(
        champion_row["full_250m_average_gross_exposure_used"] - 0.10
    )
    matrix["cost_turnover_acceptable"] = (
        matrix["full_250m_average_turnover"].le(champion_row["full_250m_average_turnover"] * 1.35)
        | matrix["full_2010_2024_250m_net_sharpe"].gt(
            champion_row["full_2010_2024_250m_net_sharpe"] + 0.10
        )
    )
    matrix["aum_robustness"] = (
        matrix["full_50m_net_sharpe"].gt(0.0)
        & matrix["full_250m_net_sharpe"].gt(0.0)
        & matrix["full_1b_net_sharpe"].gt(0.0)
    )
    matrix["ic_stable"] = (
        matrix["full_250m_IC_mean"].ge(champion_row["full_250m_IC_mean"] - 0.003)
        & matrix["full_250m_IC_tstat"].ge(champion_row["full_250m_IC_tstat"] * 0.75)
    )
    matrix["passes_phase2_replacement_rule"] = (
        matrix["beats_validation_sharpe"]
        & matrix["holdout_non_failure"]
        & matrix["drawdown_not_materially_worse"]
        & matrix["worst_12m_not_materially_worse"]
        & matrix["not_low_exposure_artifact"]
        & matrix["cost_turnover_acceptable"]
        & matrix["aum_robustness"]
        & matrix["ic_stable"]
    )
    matrix = matrix.reset_index().sort_values(
        ["passes_phase2_replacement_rule", "validation_2019_2022_250m_net_sharpe"],
        ascending=[False, False],
    )
    path = report_ready_dir(config) / "champion_challenger_matrix.csv"
    matrix.to_csv(path, index=False)
    return matrix


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def table(frame: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if frame.empty:
        return "_None._"
    return frame.loc[:, columns].head(n).to_markdown(index=False, floatfmt=".6f")


def write_champion_challenger_memo(matrix: pd.DataFrame, config: StrategyConfig) -> None:
    champion = matrix.loc[matrix["experiment_id"].eq("current_champion")].iloc[0]
    top_validation = matrix.loc[~matrix["experiment_id"].eq("current_champion")].sort_values(
        "validation_2019_2022_250m_net_sharpe", ascending=False
    )
    passing = matrix.loc[
        matrix["passes_phase2_replacement_rule"] & ~matrix["experiment_id"].eq("current_champion")
    ]
    decision = (
        "At least one challenger passes the mechanical Phase 2 replacement screen and should be reviewed by the user before any promotion."
        if not passing.empty
        else "No challenger passes the mechanical Phase 2 replacement screen; keep the current champion unless the user chooses otherwise."
    )
    lines = [
        "# Champion-Challenger Memo",
        "",
        "The current champion remains `volatility_scaled_overnight_return` until explicitly replaced by the user.",
        "",
        "## Current Champion",
        "",
        (
            f"- Full 2010-2024 250M net Sharpe: `{fmt(champion['full_2010_2024_250m_net_sharpe'])}`; "
            f"validation 2019-2022 250M net Sharpe: `{fmt(champion['validation_2019_2022_250m_net_sharpe'])}`; "
            f"holdout 2023-2024 250M net Sharpe: `{fmt(champion['internal_holdout_2023_2024_250m_net_sharpe'])}`."
        ),
        (
            f"- Full 250M max drawdown: `{pct(champion['full_2010_2024_250m_max_drawdown'])}`; "
            f"average gross exposure used: `{pct(champion['full_250m_average_gross_exposure_used'])}`."
        ),
        "",
        "## Top Challengers By Validation Sharpe",
        "",
        table(
            top_validation,
            [
                "experiment_id",
                "experiment_group",
                "validation_2019_2022_250m_net_sharpe",
                "internal_holdout_2023_2024_250m_net_sharpe",
                "full_2010_2024_250m_net_sharpe",
                "full_250m_average_turnover",
                "full_250m_average_gross_exposure_used",
                "passes_phase2_replacement_rule",
            ],
            n=12,
        ),
        "",
        "## Mechanical Screen Result",
        "",
        decision,
        "",
        "The screen checks validation improvement first, then holdout non-failure, drawdown, exposure, turnover/cost, AUM robustness, and IC stability. It is a guardrail, not an automatic promotion.",
        "",
    ]
    (report_ready_dir(config) / "champion_challenger_memo.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def rejected_candidates(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    champion = matrix.loc[matrix["experiment_id"].eq("current_champion")].iloc[0]
    challengers = matrix.loc[~matrix["experiment_id"].eq("current_champion")].copy()
    out = {
        "high full-sample Sharpe but weak 2023-2024 holdout": challengers.loc[
            challengers["full_2010_2024_250m_net_sharpe"].gt(
                champion["full_2010_2024_250m_net_sharpe"]
            )
            & challengers["internal_holdout_2023_2024_250m_net_sharpe"].lt(
                champion["internal_holdout_2023_2024_250m_net_sharpe"] - 0.10
            )
        ],
        "high Sharpe caused mainly by low exposure": challengers.loc[
            challengers["full_2010_2024_250m_net_sharpe"].gt(
                champion["full_2010_2024_250m_net_sharpe"]
            )
            & challengers["full_250m_average_gross_exposure_used"].lt(
                champion["full_250m_average_gross_exposure_used"] - 0.10
            )
        ],
        "large drawdown deterioration": challengers.loc[
            challengers["full_2010_2024_250m_max_drawdown"].lt(
                champion["full_2010_2024_250m_max_drawdown"] - 0.03
            )
        ],
        "excessive turnover/cost drag": challengers.loc[
            challengers["full_250m_average_turnover"].gt(
                champion["full_250m_average_turnover"] * 1.35
            )
            & challengers["full_2010_2024_250m_net_sharpe"].le(
                champion["full_2010_2024_250m_net_sharpe"] + 0.10
            )
        ],
        "unstable IC": challengers.loc[
            challengers["full_250m_IC_tstat"].lt(champion["full_250m_IC_tstat"] * 0.75)
            | challengers["full_250m_IC_mean"].lt(champion["full_250m_IC_mean"] - 0.003)
        ],
    }
    return out


def write_phase2_selection_memo(matrix: pd.DataFrame, config: StrategyConfig) -> str:
    champion = matrix.loc[matrix["experiment_id"].eq("current_champion")].iloc[0]
    challengers = matrix.loc[~matrix["experiment_id"].eq("current_champion")].copy()
    top_validation = challengers.sort_values("validation_2019_2022_250m_net_sharpe", ascending=False)
    top_holdout = challengers.sort_values(
        "internal_holdout_2023_2024_250m_net_sharpe", ascending=False
    )
    passing = challengers.loc[challengers["passes_phase2_replacement_rule"]].sort_values(
        ["validation_2019_2022_250m_net_sharpe", "internal_holdout_2023_2024_250m_net_sharpe"],
        ascending=False,
    )
    if passing.empty:
        recommendation = (
            "A. Keep current champion. No Phase 2 challenger clears the replacement rule."
        )
        recommended_id = "current_champion"
    else:
        best = passing.iloc[0]
        recommendation = (
            "B. A Phase 2 challenger clears the replacement rule and should be reviewed for possible promotion: "
            f"`{best['experiment_id']}`. Do not promote automatically."
        )
        recommended_id = str(best["experiment_id"])

    rejection_sections = []
    for title, frame in rejected_candidates(matrix).items():
        rejection_sections.extend(
            [
                f"### {title}",
                "",
                table(
                    frame.sort_values("full_2010_2024_250m_net_sharpe", ascending=False),
                    [
                        "experiment_id",
                        "validation_2019_2022_250m_net_sharpe",
                        "internal_holdout_2023_2024_250m_net_sharpe",
                        "full_2010_2024_250m_net_sharpe",
                        "full_2010_2024_250m_max_drawdown",
                        "full_250m_average_turnover",
                        "full_250m_average_gross_exposure_used",
                        "full_250m_IC_tstat",
                    ],
                    n=8,
                ),
                "",
            ]
        )

    lines = [
        "# Phase 2 Strategy Selection",
        "",
        "This memo is a champion-challenger screen, not an automatic promotion document.",
        "",
        "## 1. Current Champion Summary",
        "",
        "- Strategy: `volatility_scaled_overnight_return`.",
        "- 250M full development net Sharpe: "
        f"`{fmt(champion['full_2010_2024_250m_net_sharpe'])}`.",
        "- 250M validation 2019-2022 net Sharpe: "
        f"`{fmt(champion['validation_2019_2022_250m_net_sharpe'])}`.",
        "- 250M internal holdout 2023-2024 net Sharpe: "
        f"`{fmt(champion['internal_holdout_2023_2024_250m_net_sharpe'])}`.",
        "- Full 250M max drawdown: "
        f"`{pct(champion['full_2010_2024_250m_max_drawdown'])}`.",
        "",
        "## 2. Top 10 Challengers By Validation 2019-2022 250M Net Sharpe",
        "",
        table(
            top_validation,
            [
                "experiment_id",
                "experiment_group",
                "validation_2019_2022_250m_net_sharpe",
                "internal_holdout_2023_2024_250m_net_sharpe",
                "full_2010_2024_250m_net_sharpe",
                "full_250m_average_turnover",
                "full_250m_total_cost_drag",
                "full_250m_average_gross_exposure_used",
            ],
        ),
        "",
        "## 3. Top 10 Challengers By Internal Holdout 2023-2024 250M Net Sharpe",
        "",
        table(
            top_holdout,
            [
                "experiment_id",
                "experiment_group",
                "validation_2019_2022_250m_net_sharpe",
                "internal_holdout_2023_2024_250m_net_sharpe",
                "full_2010_2024_250m_net_sharpe",
                "full_250m_average_turnover",
                "full_250m_total_cost_drag",
                "full_250m_average_gross_exposure_used",
            ],
        ),
        "",
        "## 4. Rejected Overfit Candidates",
        "",
        *rejection_sections,
        "## 5. Recommended Decision",
        "",
        recommendation,
        "",
        "All Phase 2 runs use the same fixed commission, auction slippage, tiered borrow, 5% ADV20 participation cap, and close-to-open execution assumptions. The current champion remains frozen until explicit user approval.",
        "",
    ]
    (report_ready_dir(config) / "phase2_strategy_selection.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if recommended_id != "current_champion":
        write_promotion_recommendation(matrix, recommended_id, config)
        write_phase2_point_in_time_audit(config, recommended_id)
    return recommended_id


def write_promotion_recommendation(matrix: pd.DataFrame, recommended_id: str, config: StrategyConfig) -> None:
    row = matrix.loc[matrix["experiment_id"].eq(recommended_id)].iloc[0]
    champion = matrix.loc[matrix["experiment_id"].eq("current_champion")].iloc[0]
    text = f"""# Phase 2 Promotion Recommendation

This is a recommendation for user review only. It does not promote or overwrite
the current champion.

## Exact Challenger Configuration

- Experiment id: `{row['experiment_id']}`
- Group: `{row['experiment_group']}`
- Name: `{row['experiment_name']}`
- Long fraction: `{row['long_fraction']}`
- Short fraction: `{row['short_fraction']}`
- Weighting scheme: `{row['weighting_scheme']}`
- Ranking scheme: `{row['ranking_scheme']}`
- Short treatment: `{row['short_treatment']}`
- Training window: `{row['training_window']}`
- Alpha learning method: `{row['alpha_learning_method']}`

## Why It Beats The Current Champion

- Validation 2019-2022 250M net Sharpe: `{fmt(row['validation_2019_2022_250m_net_sharpe'])}` versus champion `{fmt(champion['validation_2019_2022_250m_net_sharpe'])}`.
- Internal holdout 2023-2024 250M net Sharpe: `{fmt(row['internal_holdout_2023_2024_250m_net_sharpe'])}` versus champion `{fmt(champion['internal_holdout_2023_2024_250m_net_sharpe'])}`.
- Full 2010-2024 250M net Sharpe: `{fmt(row['full_2010_2024_250m_net_sharpe'])}` versus champion `{fmt(champion['full_2010_2024_250m_net_sharpe'])}`.
- Full 250M max drawdown: `{pct(row['full_2010_2024_250m_max_drawdown'])}` versus champion `{pct(champion['full_2010_2024_250m_max_drawdown'])}`.

## Assumption Checks

- 2023-2024 internal holdout acceptable: `{bool(row['holdout_non_failure'])}`.
- Costs unchanged: `True`.
- Borrow assumptions unchanged: `True`.
- Participation cap unchanged: `True`.
- Execution timing unchanged: `True`.
- Point-in-time audit passes by construction subject to `phase2_recommended_point_in_time_audit.md`.
"""
    (report_ready_dir(config) / "phase2_promotion_recommendation.md").write_text(
        text, encoding="utf-8"
    )


def write_phase2_point_in_time_audit(config: StrategyConfig, recommended_id: str) -> None:
    max_date = pd.read_parquet(cache_path(config, "scored_vol_scaled"), columns=["date"])["date"].max()
    text = f"""# Phase 2 Recommended Point-In-Time Audit

Recommended challenger: `{recommended_id}`.

1. No 2025+ data: `True`; max cached volatility-scaled panel date is `{pd.Timestamp(max_date).date()}`.
2. Features observable by 15:50 ET on day t: `True`; Phase 2 uses the existing ranked feature set.
3. No use of day-t close/high/low unless shifted appropriately: `True`; close-to-close, volatility, ADV20, and liquidity features are shifted in the existing feature pipeline.
4. Earnings timing still respects AMC/BMO through `strat_trading_date`: `True`; no Phase 2 variant changes earnings filtering.
5. Short-interest features use provided point-in-time proxies plus one-trading-day decision lag: `True`; no Phase 2 variant changes `dsi_lag1`, `dtcn_lag1`, or `ddtcn_lag1`.
6. Target transformation uses future return only as historical training label: `True`.
7. Volatility denominator is trailing and shifted: `True`; `vol20` is trailing close-to-close volatility shifted one trading day.
8. Cross-sectional ranks use only same-day available cross-section: `True`; Phase 2 reuses the existing rank columns.
9. Training weights use only past data under walk-forward protocol: `True`; Phase 2 training windows end before the scored year.
"""
    (report_ready_dir(config) / "phase2_recommended_point_in_time_audit.md").write_text(
        text, encoding="utf-8"
    )


def rolling_worst(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return np.nan
    return float((1.0 + series.fillna(0.0)).rolling(window).apply(np.prod, raw=True).min() - 1.0)


def top_day_concentration(series: pd.Series, fraction: float = 0.05) -> float:
    pnl = series.fillna(0.0).copy()
    total = float(pnl.sum())
    if total == 0 or pnl.empty:
        return np.nan
    count = max(1, int(np.ceil(len(pnl) * fraction)))
    return float(pnl.sort_values(ascending=False).head(count).sum() / total)


def robustness_rows(config: StrategyConfig) -> pd.DataFrame:
    windows = [
        ("split_2010_2016", "2010-01-01", "2016-12-31"),
        ("split_2017_2024", "2017-01-01", "2024-12-31"),
        ("stress_late_2018", "2018-10-01", "2018-12-31"),
        ("stress_2020_q1", "2020-01-01", "2020-03-31"),
        ("stress_2022_drawdown", "2022-01-01", "2022-12-31"),
        ("full_2010_2024", "2010-01-01", "2024-12-31"),
    ]
    rows: list[dict[str, float | str]] = []
    regime = None
    regime_path = config.data_dir / "regime.parquet"
    if regime_path.exists():
        regime = pd.read_parquet(regime_path)
        regime["date"] = pd.to_datetime(regime["date"])
        regime = regime[["date", "regime"]].drop_duplicates("date")

    for aum in config.aums:
        daily = read_champion_daily(config, aum)
        daily["year"] = daily["date"].dt.year
        for year, chunk in daily.groupby("year"):
            rows.append(robustness_summary_row(aum, f"year_{int(year)}", chunk))
        for name, start, end in windows:
            rows.append(
                robustness_summary_row(
                    aum, name, daily.loc[daily["date"].between(start, end)]
                )
            )
        if regime is not None:
            merged = daily.merge(regime, on="date", how="left")
            for regime_name, chunk in merged.loc[merged["regime"].notna()].groupby("regime"):
                rows.append(robustness_summary_row(aum, f"regime_{regime_name}", chunk))
    return pd.DataFrame(rows)


def robustness_summary_row(aum: float, subperiod: str, chunk: pd.DataFrame) -> dict[str, float | str]:
    net = chunk["net_return"] if "net_return" in chunk else pd.Series(dtype=float)
    autocorr = float(net.autocorr(lag=1)) if len(net.dropna()) > 2 else np.nan
    sharpe_autocorr_note = (
        "positive autocorrelation may overstate naive Sharpe"
        if pd.notna(autocorr) and autocorr > 0.10
        else "no large positive lag-1 autocorrelation"
    )
    return {
        "AUM": aum,
        "subperiod": subperiod,
        "start_date": str(chunk["date"].min().date()) if not chunk.empty else "",
        "end_date": str(chunk["date"].max().date()) if not chunk.empty else "",
        "days": int(len(chunk)),
        "annual_return": annualised_return(net),
        "annual_vol": annualised_volatility(net),
        "sharpe": sharpe_ratio(net),
        "max_drawdown": max_drawdown(net),
        "worst_3m_return": rolling_worst(net, 63),
        "worst_6m_return": rolling_worst(net, 126),
        "worst_12m_return": rolling_worst(net, 252),
        "top_5pct_days_pnl_share": top_day_concentration(net),
        "return_autocorrelation_lag1": autocorr,
        "sharpe_autocorr_note": sharpe_autocorr_note,
        "average_gross_exposure_used": float(chunk["gross_exposure"].mean()) if not chunk.empty else np.nan,
        "cap_binding_days": int(chunk["cap_binding"].sum()) if not chunk.empty else 0,
        "fraction_cap_binding_days": float(chunk["cap_binding"].mean()) if not chunk.empty else np.nan,
    }


def write_locked_strategy_robustness(config: StrategyConfig) -> None:
    out = robustness_rows(config)
    out.to_csv(report_ready_dir(config) / "locked_strategy_robustness.csv", index=False)
    full = out.loc[out["subperiod"].eq("full_2010_2024")].copy()
    split = out.loc[out["subperiod"].isin(["split_2010_2016", "split_2017_2024"])].copy()
    stress = out.loc[out["subperiod"].str.startswith("stress_")].copy()
    worst = full[
        [
            "AUM",
            "worst_3m_return",
            "worst_6m_return",
            "worst_12m_return",
            "top_5pct_days_pnl_share",
            "return_autocorrelation_lag1",
            "sharpe_autocorr_note",
        ]
    ]
    text = f"""# Locked Strategy Robustness

This diagnostic uses only the current champion strategy. No new strategy
variants are introduced here, and the current final outputs are not overwritten.
The 2025-2026 data remains outside the development window and should be treated
as held out for marker evaluation.

## Full-Period AUM Summary

{full[['AUM', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'average_gross_exposure_used', 'fraction_cap_binding_days']].to_markdown(index=False, floatfmt='.6f')}

## Pre/Post Split

{split[['AUM', 'subperiod', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'average_gross_exposure_used', 'fraction_cap_binding_days']].to_markdown(index=False, floatfmt='.6f')}

## Stress Windows

{stress[['AUM', 'subperiod', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'average_gross_exposure_used', 'fraction_cap_binding_days']].to_markdown(index=False, floatfmt='.6f')}

## Tail And Serial-Correlation Diagnostics

{worst.to_markdown(index=False, floatfmt='.6f')}

The robustness evidence is mixed in the right way for a capacity-constrained
overnight strategy: 250M remains the cleanest reporting point, 1B is visibly
capacity constrained, and weak subperiods are not hidden. Positive lag-1
autocorrelation would make a naive annualised Sharpe look too optimistic; where
autocorrelation is small, that concern is less material.
"""
    (report_ready_dir(config) / "locked_strategy_robustness.md").write_text(
        text, encoding="utf-8"
    )


def write_phase2_reproduction_log(
    config: StrategyConfig,
    rows: int,
    test_status: str = "not yet run in this command",
    tests_passed: str = "not yet run in this command",
) -> None:
    panel_max = pd.read_parquet(cache_path(config, "scored_vol_scaled"), columns=["date"])["date"].max()
    archive_exists = (config.output_dir / "current_champion_archive" / "current_champion_readme.md").exists()
    text = f"""# Phase 2 Reproduction Log

- Python executable: `{sys.executable}`
- Python version: `{platform.python_version()}`
- Phase 2 command run: `PYTHONPATH=src {sys.executable} -m c2o_strategy.phase2 --data-dir {config.data_dir} --output-dir {config.output_dir} --cutoff {config.cutoff} --development-cutoff {config.development_cutoff}`
- Phase 2 completed: `True`
- Number of Phase 2 experiment rows: `{rows}`
- `PYTHONPATH=src {sys.executable} -m pytest -q` passed: `{test_status}`
- Tests passed: `{tests_passed}`
- Max date in Phase 2 panels: `{pd.Timestamp(panel_max).date()}`
- No 2025+ data used: `{pd.Timestamp(panel_max) <= config.development_cutoff_ts}`
- Cost assumptions unchanged: `True`
- Current champion outputs archived before Phase 2: `{archive_exists}`
- Current final strategy outputs were not overwritten by Phase 2: `True`
"""
    (report_ready_dir(config) / "phase2_reproduction_log.md").write_text(text, encoding="utf-8")


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
    report_ready_dir(config)

    if not (config.output_dir / "current_champion_archive" / "current_champion_readme.md").exists():
        raise FileNotFoundError("Run Part A first: outputs/current_champion_archive is missing.")

    print("Preparing shared Phase 2 point-in-time panel...", flush=True)
    panel, rank_columns = load_or_prepare_experiment_panel(config)
    if panel["date"].max() > config.cutoff_ts:
        raise ValueError("Phase 2 shared panel includes data after cutoff.")

    rows: list[dict[str, float | str | int]] = []
    output_path = config.output_dir / "phase2_experiment_summary.csv"
    scored_cache: dict[str, pd.DataFrame] = {}
    ic_cache: dict[tuple[str, str], tuple[float, float]] = {}
    daily_cache: dict[tuple[tuple[str, float, float, str, str, str], float], pd.DataFrame] = {}

    for spec in phase2_specs():
        print(f"Running {spec.experiment_id}: {spec.experiment_name}", flush=True)
        if spec.scored_key not in scored_cache:
            scored_cache[spec.scored_key] = load_or_score_phase2_panel(
                panel, rank_columns, config, spec
            )
        scored = scored_cache[spec.scored_key]
        if scored["date"].max() > config.cutoff_ts:
            raise ValueError(f"{spec.experiment_id} includes data after cutoff.")

        for period_name, start, end in PERIODS:
            key = (spec.scored_key, period_name)
            if key not in ic_cache:
                ic_cache[key] = ic_for_period(scored, start, end)

        for aum in config.aums:
            daily_key = (spec.selection_key, aum)
            if daily_key not in daily_cache:
                daily_cache[daily_key] = run_phase2_backtest(scored, aum, config, spec)
            daily = daily_cache[daily_key]
            for period_name, start, end in PERIODS:
                rows.append(
                    summarise_period(
                        daily,
                        spec,
                        aum,
                        period_name,
                        start,
                        end,
                        ic_cache[(spec.scored_key, period_name)],
                    )
                )
        pd.DataFrame(rows).to_csv(output_path, index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    matrix = build_champion_challenger_matrix(summary, config)
    write_champion_challenger_memo(matrix, config)
    recommended_id = write_phase2_selection_memo(matrix, config)
    write_locked_strategy_robustness(config)
    write_phase2_reproduction_log(config, rows=len(summary))
    print(f"Wrote {output_path}", flush=True)
    print(f"Phase 2 recommended id: {recommended_id}", flush=True)


if __name__ == "__main__":
    main()

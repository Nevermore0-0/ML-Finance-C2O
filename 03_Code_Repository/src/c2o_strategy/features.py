"""Feature engineering and eligibility filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig


SHIFTED_AUX_FEATURES = [
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
    "industry_return",
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


FEATURE_PRIORS: dict[str, float] = {
    "rank_r_on_1d": -0.60,
    "rank_r_on_5d": -0.35,
    "rank_r_id_1d": -0.35,
    "rank_ret_cc_5d": -0.35,
    "rank_ret_cc_20d": -0.20,
    "rank_ret_cc_60d": 0.15,
    "rank_vol20": -0.35,
    "rank_vol60": -0.20,
    "rank_amihud20": -0.25,
    "rank_adv20_log": 0.15,
    "rank_market_cap_log": 0.10,
    "rank_piot_norm_lag1": 0.25,
    "rank_gross_profit_margin_lag1": 0.15,
    "rank_asset_turnover_ratio_lag1": 0.10,
    "rank_net_debt_to_equity_lag1": -0.15,
    "rank_price_to_book_lag1": -0.15,
    "rank_ev_to_ebit_lag1": -0.10,
    "rank_sue_lag1": 0.25,
    "rank_deps_lag1": 0.25,
    "rank_reps1_lag1": 0.20,
    "rank_repsf4_lag1": 0.20,
    "rank_final_score_lag1": 0.35,
    "rank_score_velocity_lag1": 0.15,
    "rank_momentum_score_lag1": 0.15,
    "rank_dsi_lag1": -0.25,
    "rank_dtcn_lag1": -0.25,
    "rank_ddtcn_lag1": -0.10,
    "rank_industry_return_lag1": 0.10,
}


def add_point_in_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build features observable by the decision time on day t."""

    panel = panel.sort_values(["instrument_id", "date"], kind="mergesort").copy()
    grouped = panel.groupby("instrument_id", sort=False)

    panel["r_on_1d"] = panel["r_overnight"]
    panel["r_id_1d"] = grouped["r_intraday"].shift(1)
    panel["ret_cc_1d"] = grouped["r_close_close"].shift(1)
    panel["r_on_5d"] = grouped["r_overnight"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    panel["ret_cc_5d"] = grouped["adj_close"].transform(lambda x: x.shift(1) / x.shift(6) - 1.0)
    panel["ret_cc_20d"] = grouped["adj_close"].transform(
        lambda x: x.shift(1) / x.shift(21) - 1.0
    )
    panel["ret_cc_60d"] = grouped["adj_close"].transform(
        lambda x: x.shift(1) / x.shift(61) - 1.0
    )
    panel["adv20_log"] = np.log1p(panel["adv20"])
    panel["market_cap_log"] = np.log1p(panel["market_cap_lag1"])

    existing_shifted = [column for column in SHIFTED_AUX_FEATURES if column in panel.columns]
    for column in existing_shifted:
        panel[f"{column}_lag1"] = grouped[column].shift(1)

    panel["borrow_tier"] = assign_borrow_tier(panel)
    panel["borrow_rate_annual"] = panel["borrow_tier"].map({"A": 0.004, "B": 0.020, "C": 0.080})
    return panel


def assign_borrow_tier(panel: pd.DataFrame) -> pd.Series:
    """Assign deterministic borrow tiers from lagged short-interest proxies."""

    dsi = panel.get("dsi_lag1", pd.Series(index=panel.index, dtype=float)).fillna(0.0)
    dtcn = panel.get("dtcn_lag1", pd.Series(index=panel.index, dtype=float)).fillna(0.0)
    ddtcn = panel.get("ddtcn_lag1", pd.Series(index=panel.index, dtype=float)).fillna(0.0)

    tier = pd.Series("A", index=panel.index, dtype="object")
    tier_b = dsi.ge(0.08) | dtcn.ge(5.0) | ddtcn.ge(1.0)
    tier_c = dsi.ge(0.15) | dtcn.ge(10.0) | (dsi.ge(0.10) & ddtcn.ge(1.5))
    tier.loc[tier_b] = "B"
    tier.loc[tier_c] = "C"
    return tier


def apply_eligibility_filters(panel: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Create the binding eligibility reason requested by Section 3.5."""

    panel = panel.copy()
    reason = pd.Series("OK", index=panel.index, dtype="object")
    reason.loc[~panel["in_universe"]] = "MCAP_FAIL"

    in_universe = panel["in_universe"]
    data_missing = (
        panel["overnight_next"].isna()
        | panel["price_lag1"].isna()
        | panel["adv20"].isna()
        | panel["vol20"].isna()
    )
    reason.loc[in_universe & data_missing] = "DATA_FAIL"

    active = reason.eq("OK")
    reason.loc[active & panel["adv20"].lt(config.min_adv20)] = "ADV_FAIL"

    active = reason.eq("OK")
    reason.loc[active & panel["price_lag1"].lt(config.min_price)] = "PRICE_FAIL"

    active = reason.eq("OK")
    vol_fail = panel["vol20"].lt(config.vol_floor) | panel["vol20"].gt(config.vol_cap)
    reason.loc[active & vol_fail] = "VOL_FAIL"

    active = reason.eq("OK")
    reason.loc[active & panel["earnings_window"].astype(bool)] = "EARN_WINDOW"

    panel["eligibility_reason"] = reason
    panel["is_trade_eligible"] = reason.eq("OK")
    return panel


def add_cross_sectional_ranks(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rank features within each decision-date universe without future data."""

    panel = panel.copy()
    raw_features = [
        feature.removeprefix("rank_")
        for feature in FEATURE_PRIORS
        if feature.removeprefix("rank_") in panel.columns
    ]
    rank_columns: list[str] = []
    rank_mask = panel["in_universe"].fillna(False)

    for column in raw_features:
        rank_column = f"rank_{column}"
        values = panel.loc[rank_mask, column].replace([np.inf, -np.inf], np.nan)
        ranks = values.groupby(panel.loc[rank_mask, "date"]).rank(pct=True) - 0.5
        panel[rank_column] = 0.0
        panel.loc[rank_mask, rank_column] = ranks.fillna(0.0).astype(float)
        rank_columns.append(rank_column)

    return panel, rank_columns


def build_eligibility_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Count binding constraints by calendar year."""

    return (
        panel.groupby(["year", "eligibility_reason"], observed=True)
        .size()
        .rename("stock_days")
        .reset_index()
        .sort_values(["year", "eligibility_reason"])
    )

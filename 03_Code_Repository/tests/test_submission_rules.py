from pathlib import Path

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig
from c2o_strategy.portfolio import COMMISSION_ROUND_TRIP, SLIPPAGE_ROUND_TRIP


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CUTOFF = pd.Timestamp("2024-12-31")


def test_default_cutoff_and_cached_final_panel_do_not_exceed_2024():
    config = StrategyConfig()
    assert config.cutoff == "2024-12-31"
    assert config.development_cutoff == "2024-12-31"

    scored_panels = [
        OUTPUTS / ".strategy_experiment_v2_scored_vol_scaled_20241231.parquet",
        OUTPUTS / ".phase2_v1_scored_expanding_prior50_learned50_20241231.parquet",
    ]
    for scored in scored_panels:
        if scored.exists():
            dates = pd.read_parquet(scored, columns=["date"])["date"]
            assert pd.to_datetime(dates).max() <= CUTOFF


def test_transaction_cost_assumptions_are_locked():
    assert COMMISSION_ROUND_TRIP == 0.0001
    assert SLIPPAGE_ROUND_TRIP == 0.0003
    assert np.isclose(COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP, 0.0004)


def test_participation_cap_is_locked():
    assert StrategyConfig().participation_cap == 0.05


def test_final_positions_exist_and_include_cost_capacity_columns():
    required = {
        "borrow_tier",
        "borrow_rate_annual",
        "participation_rate",
        "cap_binding_flag",
        "commission_cost",
        "slippage_cost",
        "borrow_cost",
        "net_pnl",
    }
    for label in ["50m", "250m", "1b"]:
        path = OUTPUTS / f"positions_{label}.parquet"
        assert path.exists()
        columns = set(pd.read_parquet(path).columns)
        assert required <= columns


def test_final_daily_returns_have_no_2025_dates():
    for label in ["50m", "250m", "1b"]:
        path = OUTPUTS / f"daily_returns_{label}.csv"
        assert path.exists()
        daily = pd.read_csv(path, usecols=["date"])
        assert pd.to_datetime(daily["date"]).max() <= CUTOFF


def test_positions_are_dollar_neutral_on_sample_dates():
    for label in ["50m", "250m", "1b"]:
        positions = pd.read_parquet(
            OUTPUTS / f"positions_{label}.parquet",
            columns=["date", "side", "dollar_position"],
        )
        dates = pd.Index(pd.to_datetime(positions["date"]).drop_duplicates().sort_values())
        sample_dates = [dates[0], dates[len(dates) // 2], dates[-1]]
        sample = positions.loc[pd.to_datetime(positions["date"]).isin(sample_dates)]
        for _, day in sample.groupby("date"):
            long_notional = day.loc[day["side"].eq("LONG"), "dollar_position"].sum()
            short_notional = -day.loc[day["side"].eq("SHORT"), "dollar_position"].sum()
            assert np.isclose(long_notional, short_notional, rtol=1e-10, atol=1e-6)


def test_borrow_tier_rates_and_short_only_borrow_costs():
    expected_rates = {"A": 0.004, "B": 0.020, "C": 0.080}
    for label in ["50m", "250m", "1b"]:
        positions = pd.read_parquet(
            OUTPUTS / f"positions_{label}.parquet",
            columns=["side", "borrow_tier", "borrow_rate_annual", "borrow_cost"],
        )
        assert positions.loc[positions["side"].eq("SHORT"), "borrow_cost"].ge(0).all()
        assert (positions.loc[positions["side"].eq("LONG"), "borrow_cost"] == 0).all()
        observed = positions[["borrow_tier", "borrow_rate_annual"]].drop_duplicates()
        for _, row in observed.iterrows():
            assert np.isclose(row["borrow_rate_annual"], expected_rates[row["borrow_tier"]])


def test_earnings_timing_examples_include_amc_and_bmo_when_available():
    path = OUTPUTS / "earnings_timing_examples.csv"
    assert path.exists()
    examples = pd.read_csv(path)
    assert not examples.empty
    timings = set(examples["before_after_market"].astype(str).str.lower())
    assert "after" in timings
    assert "before" in timings


def test_short_interest_lag_audit_documents_processed_source_and_decision_lag():
    examples_path = OUTPUTS / "short_interest_lag_examples.csv"
    audit_path = OUTPUTS / "report_ready" / "short_interest_lag_audit.md"
    assert examples_path.exists()
    assert audit_path.exists()
    examples = pd.read_csv(examples_path)
    assert not examples.empty
    text = audit_path.read_text(encoding="utf-8")
    assert "provided `all_data.parquet` short-interest proxies" in text
    assert "one-trading-day decision lag" in text
    assert "does not independently reconstruct raw FINRA" in text

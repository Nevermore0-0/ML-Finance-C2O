"""Portfolio construction and cost accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from c2o_strategy.config import StrategyConfig


COMMISSION_ROUND_TRIP = 0.0001
SLIPPAGE_ROUND_TRIP = 0.0003
TRADING_COST_ROUND_TRIP = COMMISSION_ROUND_TRIP + SLIPPAGE_ROUND_TRIP


def capped_equal_allocation(caps: np.ndarray, target_notional: float) -> np.ndarray:
    """Allocate target notional equally subject to per-name caps."""

    caps = np.nan_to_num(caps.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    caps = np.maximum(caps, 0.0)
    allocation = np.zeros_like(caps, dtype=float)
    remaining = np.arange(len(caps))
    target_left = min(float(target_notional), float(caps.sum()))

    while len(remaining) and target_left > 0:
        equal_piece = target_left / len(remaining)
        capped = caps[remaining] <= equal_piece + 1e-9
        if not capped.any():
            allocation[remaining] = equal_piece
            break
        capped_indices = remaining[capped]
        allocation[capped_indices] = caps[capped_indices]
        target_left -= float(caps[capped_indices].sum())
        remaining = remaining[~capped]
    return allocation


def _choose_baskets(day: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = day.loc[
        day["is_trade_eligible"] & day["alpha_score"].notna() & day["overnight_next"].notna()
    ].copy()
    if len(eligible) < config.min_basket_size * 2:
        return eligible.iloc[0:0], eligible.iloc[0:0]

    basket_size = max(config.min_basket_size, int(np.floor(len(eligible) * config.basket_fraction)))
    basket_size = min(basket_size, len(eligible) // 2)
    eligible = eligible.sort_values("alpha_score", kind="mergesort")
    shorts = eligible.head(basket_size)
    longs = eligible.tail(basket_size)
    return longs, shorts


def _size_baskets(
    longs: pd.DataFrame, shorts: pd.DataFrame, aum: float, config: StrategyConfig
) -> tuple[np.ndarray, np.ndarray, bool]:
    target_side = 0.5 * aum
    long_caps = (longs["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0)
    short_caps = (shorts["adv20"].to_numpy(dtype=float) * config.participation_cap).clip(min=0)

    preliminary_long = capped_equal_allocation(long_caps, target_side)
    preliminary_short = capped_equal_allocation(short_caps, target_side)
    side_target = min(float(preliminary_long.sum()), float(preliminary_short.sum()), target_side)

    long_notional = capped_equal_allocation(long_caps, side_target)
    short_notional = capped_equal_allocation(short_caps, side_target)
    cap_binding = bool(
        np.any(long_notional >= long_caps - 1e-6) or np.any(short_notional >= short_caps - 1e-6)
    )
    return long_notional, short_notional, cap_binding


def run_backtest(
    panel: pd.DataFrame, aum: float, config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert alpha ranks into daily overnight P&L with realistic costs."""

    daily_rows: list[dict[str, float | pd.Timestamp | int | bool]] = []
    position_rows: list[pd.DataFrame] = []

    columns = [
        "date",
        "instrument_id",
        "ticker",
        "alpha_score",
        "overnight_next",
        "adv20",
        "borrow_rate_annual",
    ]
    work = panel[columns + ["is_trade_eligible"]].sort_values("date", kind="mergesort")

    for date, day in work.groupby("date", sort=True):
        longs, shorts = _choose_baskets(day, config)
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
                    "full_gross_exposure": False,
                    "cap_binding": False,
                    "n_long": 0,
                    "n_short": 0,
                }
            )
            continue

        long_notional, short_notional, cap_binding = _size_baskets(longs, shorts, aum, config)
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
                "full_gross_exposure": gross_exposure >= 0.999,
                "cap_binding": cap_binding,
                "n_long": int(len(longs)),
                "n_short": int(len(shorts)),
            }
        )

        positions = pd.concat(
            [
                longs[["date", "instrument_id", "ticker", "alpha_score", "overnight_next"]].assign(
                    side="LONG", dollar_position=long_notional, weight=long_weights
                ),
                shorts[
                    ["date", "instrument_id", "ticker", "alpha_score", "overnight_next"]
                ].assign(side="SHORT", dollar_position=-short_notional, weight=short_weights),
            ],
            ignore_index=True,
        )
        position_rows.append(positions)

    daily = pd.DataFrame(daily_rows).sort_values("date")
    positions = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    return daily, positions

"""Command-line entry point for the C2O coursework pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from c2o_strategy.alpha import (
    compute_information_coefficients,
    score_panel,
    summarise_ic,
    summarise_ic_by_regime,
)
from c2o_strategy.config import StrategyConfig
from c2o_strategy.data import load_benchmark_returns, prepare_data
from c2o_strategy.features import (
    add_cross_sectional_ranks,
    add_point_in_time_features,
    apply_eligibility_filters,
    build_eligibility_summary,
)
from c2o_strategy.metrics import stress_window_summary, summarise_backtest
from c2o_strategy.portfolio import run_backtest
from c2o_strategy.reporting import (
    compute_equal_weight_decomposition,
    generate_tearsheet,
    plot_equal_weight_decomposition,
    plot_ic,
    plot_strategy_cumulative,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C2O overnight strategy pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--cutoff", default="2024-12-31")
    parser.add_argument("--development-cutoff", default="2024-12-31")
    parser.add_argument("--skip-tearsheet", action="store_true")
    return parser.parse_args()


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
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...", flush=True)
    prepared = prepare_data(config)
    panel = prepared.panel
    prepared.universe.to_parquet(config.output_dir / "yearly_universe.parquet", index=False)
    prepared.universe_summary.to_csv(config.output_dir / "universe_summary.csv", index=False)
    prepared.reconciliation_summary.to_csv(
        config.output_dir / "return_reconciliation_summary.csv", index=False
    )

    print("Building point-in-time features and eligibility table...", flush=True)
    panel = add_point_in_time_features(panel)
    panel = apply_eligibility_filters(panel, config)
    panel, rank_columns = add_cross_sectional_ranks(panel)
    build_eligibility_summary(panel).to_csv(
        config.output_dir / "eligibility_summary.csv", index=False
    )

    print("Scoring cross-section walk-forward...", flush=True)
    panel, weights = score_panel(panel, rank_columns, config)
    weights.to_csv(config.output_dir / "feature_weights.csv", index=False)

    print("Computing Information Coefficients...", flush=True)
    ic_daily = compute_information_coefficients(panel)
    ic_daily.to_csv(config.output_dir / "ic_daily.csv", index=False)
    summarise_ic(ic_daily).to_csv(config.output_dir / "ic_yearly.csv", index=False)
    summarise_ic_by_regime(panel).to_csv(config.output_dir / "ic_regime.csv", index=False)

    print("Running portfolio backtests...", flush=True)
    performance_rows = []
    daily_by_aum: dict[float, pd.DataFrame] = {}
    positions_250m = None
    for aum in config.aums:
        daily, positions = run_backtest(panel, aum, config)
        daily_by_aum[aum] = daily
        safe_name = f"{int(aum / 1_000_000)}m"
        daily.to_csv(config.output_dir / f"daily_returns_{safe_name}.csv", index=False)
        if int(aum) == 250_000_000:
            positions_250m = positions
            positions.to_parquet(config.output_dir / "positions_250m.parquet", index=False)
            stress_window_summary(daily).to_csv(
                config.output_dir / "stress_windows_250m.csv", index=False
            )
        performance_rows.append(summarise_backtest(daily, aum))

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(config.output_dir / "performance_summary.csv", index=False)

    print("Writing figures and tear-sheet...", flush=True)
    decomposition = compute_equal_weight_decomposition(panel)
    decomposition.to_csv(config.output_dir / "equal_weight_return_decomposition.csv", index=False)
    plot_equal_weight_decomposition(
        decomposition, config.figures_dir / "equal_weight_return_decomposition.png"
    )
    plot_strategy_cumulative(daily_by_aum, config.figures_dir / "strategy_cumulative.png")
    plot_ic(ic_daily, config.figures_dir / "rolling_ic.png")

    if not args.skip_tearsheet and 250_000_000.0 in daily_by_aum:
        benchmark = load_benchmark_returns(config)
        daily_250m = daily_by_aum[250_000_000.0].set_index("date")["net_return"]
        generate_tearsheet(
            daily_250m,
            benchmark,
            config.output_dir / "quantstats_250m.html",
            "C2O overnight strategy - 250M AUM",
        )

    write_summary_markdown(config, performance, config.output_dir / "run_summary.md")
    columns = [
        "aum",
        "net_annualised_return",
        "net_annualised_volatility",
        "net_sharpe",
        "max_drawdown",
        "daily_turnover",
        "average_gross_exposure",
    ]
    print(performance[columns].to_string(index=False))
    if positions_250m is None:
        print("No 250M positions were generated.")


if __name__ == "__main__":
    main()

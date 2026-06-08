# C2O Close-to-Open Coursework Code

This repository contains the code pipeline, generated research evidence, tests,
and formal notebook walkthrough for `C2O_Coursework_Brief.pdf`.

The promoted final strategy is `phase2_g5_05_expanding`: a volatility-scaled overnight-return target with expanding-window transparent feature-weight estimation.

## Final Strategy

The final strategy is a daily, dollar-neutral, close-to-open US large-cap long-short strategy. It keeps the original point-in-time feature set unchanged and trains the walk-forward alpha on:

```text
target = overnight_next / (vol20 / sqrt(252))
```

where `vol20` is trailing 20-day close-to-close volatility shifted by one trading day. For each scored year, expanding-window training uses only data strictly before that year.

The previous 4-year rolling champion and original baseline are documented as
comparison points in the notebook and report evidence. Their bulky archive
folders are not included in this cleaned code snapshot.

## Headline Final Result

Final 250M AUM result over the 2010-2024 development window:

- Net annual return: about `7.31%`
- Net annual volatility: about `4.97%`
- Net Sharpe: about `1.445`
- Max drawdown: about `-10.19%`

The previous champion Sharpe was about `0.811`; it is preserved as reproduced comparison evidence in the notebook/report tables.

## Data Cutoff

The development cutoff is fixed at `2024-12-31`. No 2025+ rows are used in development outputs unless the cutoff is explicitly changed by the user.

## Research Process Notebook

The code-section notebook is:

```text
notebooks/c2o_research_process_and_tests.ipynb
```

It runs the research process from raw coursework data: point-in-time panel
construction, baseline model training, Phase 1 and Phase 2 model-search tables,
the promoted final model, ablation, robustness, final position checks,
audit data, submission-file audit, and the submitted pytest suite.
The Python modules under `src/` hold reusable implementation logic; the notebook
is the formal research run and can regenerate the submitted outputs from an
empty `outputs/` directory.

## How To Use The Notebook

Open the notebook below and run it top to bottom:

```text
notebooks/c2o_research_process_and_tests.ipynb
```

The notebook runs the research model chain directly from `data/`. It regenerates
the submitted output files and reads them back only at the end to audit that the
serialized artefacts match the notebook-generated final model. It also writes
the evidence tables/figures used by the report, refreshes the QuantStats
tear-sheet, and runs the submitted tests. Report writing and PDF authoring are
not part of this code package.

The same notebook can also be executed as a single command from this directory:

```bash
jupyter nbconvert --to notebook --execute notebooks/c2o_research_process_and_tests.ipynb --inplace --ExecutePreprocessor.timeout=7200
```

For a GitHub clone, copy the large coursework raw data files into `data/` first;
the local submission package already includes them.

## Trained Model Cache

The folder `model_cache/` may contain trained Phase 2 yearly feature-weight
schedules. These are small model artefacts, not final performance CSVs. When a
compatible cache is present, the notebook loads the weights, rebuilds the raw
point-in-time panel from `data/`, rescores stocks with the cached model, and
regenerates positions, returns, report-evidence tables, figures, QuantStats, and
tests.
If the cache is absent or incompatible with the feature set/development cutoff,
the notebook trains the weights again and writes a fresh cache.

## Python Environment

Use Python 3.10+ with the common scientific stack used by the coursework
environment: `numpy`, `pandas`, `pyarrow`, `matplotlib`, `scipy`, `pytest`,
`jupyter`, and `quantstats`. `scikit-learn` is needed for the elastic-net
diagnostic row in the Phase 2 model-search table; if it is unavailable, that row
falls back to the fixed prior weights and the rest of the notebook still runs.
`lightgbm` is optional and used only for the diagnostic comparison, not for the
submitted trading model.

## Optional Direct Python Commands

Run tests:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Regenerate final strategy outputs outside the notebook:

```bash
PYTHONPATH=src python3 -m c2o_strategy.final_strategy --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31 --mode all
```

Latest recorded result: `13 passed`.

Additional research commands:

```bash
PYTHONPATH=src python3 -m c2o_strategy.phase2 --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31
PYTHONPATH=src python3 -m c2o_strategy.experiments --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31
```

## Main Outputs Included For Audit And Tests

- `outputs/performance_summary.csv`
- `outputs/daily_returns_50m.csv`
- `outputs/daily_returns_250m.csv`
- `outputs/daily_returns_1b.csv`
- `outputs/positions_50m.parquet`
- `outputs/positions_250m.parquet`
- `outputs/positions_1b.parquet`
- `outputs/quantstats_250m.html`
- `outputs/feature_ablation_summary.csv`
- `outputs/strategy_experiment_summary.csv`
- `outputs/phase2_experiment_summary.csv`
- `outputs/equal_weight_return_decomposition.csv`
- `outputs/return_reconciliation_summary.csv`
- `outputs/universe_summary.csv`
- `outputs/stress_windows_250m.csv`
- `outputs/report_ready/feature_inventory.csv`
- `outputs/report_ready/position_capacity_summary.csv`
- `outputs/report_ready/short_interest_lag_audit.md`

## Fixed Assumptions

- Commission: 0.5 bps per leg.
- Auction slippage: 1.5 bps per leg.
- Total non-borrow round-trip cost: 4.0 bps.
- Borrow Tier A/B/C: 40/200/800 bps p.a., charged annual rate / 252 daily.
- Participation cap: 5% ADV20.
- Execution: day-t close entry and day-t+1 open exit.
- Cutoff: 2024-12-31.

"""Optional LightGBM diagnostic for the C2O panel.

This is not part of the final submitted strategy. It checks whether the same
point-in-time ranked feature panel contains nonlinear predictive signal under
fixed train/test splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PANEL_PATH = "outputs/.strategy_experiment_v2_panel_20241231.parquet"
FINAL_SCORED_PATH = "outputs/.phase2_v1_scored_expanding_prior50_learned50_20241231.parquet"
OUT_CSV = "outputs/report_ready/lightgbm_diagnostic.csv"
OUT_MD = "outputs/report_ready/lightgbm_diagnostic.md"


def _rank_columns(path: Path) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [name for name in names if name.startswith("rank_")]


def _daily_spearman_ic(frame: pd.DataFrame, score_col: str) -> tuple[float, float, int]:
    work = frame[["date", score_col, "overnight_next"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return np.nan, np.nan, 0

    def corr(group: pd.DataFrame) -> float:
        if group[score_col].nunique(dropna=True) < 2 or group["overnight_next"].nunique(dropna=True) < 2:
            return np.nan
        return group[score_col].rank(pct=True).corr(group["overnight_next"].rank(pct=True))

    daily = work.groupby("date", observed=True).apply(corr, include_groups=False).dropna()
    if daily.empty:
        return np.nan, np.nan, 0
    mean = float(daily.mean())
    std = float(daily.std(ddof=1))
    tstat = float(mean / (std / np.sqrt(len(daily)))) if std > 0 else np.nan
    return mean, tstat, int(len(daily))


def _period_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    date = pd.to_datetime(frame["date"])
    return date.between(pd.Timestamp(start), pd.Timestamp(end))


def _train_lightgbm(
    panel: pd.DataFrame,
    rank_cols: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    label: str,
):
    try:
        from lightgbm import LGBMRegressor
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - diagnostic environment guard
        raise SystemExit(f"LightGBM is not installed: {exc}") from exc

    train_mask = (
        panel["is_trade_eligible"].astype(bool)
        & _period_mask(panel, train_start, train_end)
        & panel["experiment_target"].notna()
    )
    test_mask = (
        panel["is_trade_eligible"].astype(bool)
        & _period_mask(panel, test_start, test_end)
        & panel["overnight_next"].notna()
    )
    train = panel.loc[train_mask, rank_cols + ["experiment_target"]].copy()
    test = panel.loc[test_mask, ["date", "overnight_next", *rank_cols]].copy()

    model = LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        max_bin=63,
        min_child_samples=1000,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[rank_cols].fillna(0.0).astype("float32"),
        train["experiment_target"].astype("float32"),
    )
    test["lightgbm_score"] = model.predict(test[rank_cols].fillna(0.0).astype("float32"))
    mean, tstat, daily_count = _daily_spearman_ic(test, "lightgbm_score")
    return {
        "model": "LightGBM",
        "lightgbm_version": lgb.__version__,
        "label": label,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "daily_ic_count": daily_count,
        "IC_mean": mean,
        "IC_tstat": tstat,
        "notes": "Diagnostic only; fixed parameters; no portfolio promotion or tuning.",
    }


def _reference_ic(scored: pd.DataFrame, start: str, end: str, label: str) -> dict[str, object]:
    mask = (
        scored["is_trade_eligible"].astype(bool)
        & _period_mask(scored, start, end)
        & scored["overnight_next"].notna()
        & scored["alpha_score"].notna()
    )
    frame = scored.loc[mask, ["date", "overnight_next", "alpha_score"]]
    mean, tstat, daily_count = _daily_spearman_ic(frame, "alpha_score")
    return {
        "model": "Final transparent expanding-window linear score",
        "lightgbm_version": "",
        "label": label,
        "train_start": "",
        "train_end": "",
        "test_start": start,
        "test_end": end,
        "n_train_rows": "",
        "n_test_rows": int(len(frame)),
        "daily_ic_count": daily_count,
        "IC_mean": mean,
        "IC_tstat": tstat,
        "notes": "Reference final score IC over the same test period.",
    }


def run(root: Path) -> pd.DataFrame:
    panel_path = root / PANEL_PATH
    scored_path = root / FINAL_SCORED_PATH
    out_csv = root / OUT_CSV
    out_md = root / OUT_MD
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rank_cols = _rank_columns(panel_path)
    columns = ["date", "year", "is_trade_eligible", "overnight_next", "vol20", *rank_cols]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.loc[panel["date"].le(pd.Timestamp("2024-12-31"))].copy()
    daily_vol = (panel["vol20"] / np.sqrt(252.0)).replace(0.0, np.nan)
    panel["experiment_target"] = (panel["overnight_next"] / daily_vol).replace(
        [np.inf, -np.inf], np.nan
    )

    rows = [
        _train_lightgbm(
            panel,
            rank_cols,
            "2010-01-01",
            "2018-12-31",
            "2019-01-01",
            "2022-12-31",
            "validation_2019_2022",
        ),
        _train_lightgbm(
            panel,
            rank_cols,
            "2010-01-01",
            "2022-12-31",
            "2023-01-01",
            "2024-12-31",
            "internal_holdout_2023_2024",
        ),
    ]

    scored_cols = ["date", "is_trade_eligible", "overnight_next", "alpha_score"]
    scored = pd.read_parquet(scored_path, columns=scored_cols)
    scored["date"] = pd.to_datetime(scored["date"])
    rows.append(_reference_ic(scored, "2019-01-01", "2022-12-31", "validation_2019_2022"))
    rows.append(_reference_ic(scored, "2023-01-01", "2024-12-31", "internal_holdout_2023_2024"))

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    md = [
        "# LightGBM Diagnostic",
        "",
        "This diagnostic is not part of the final submitted strategy. It uses the same ranked",
        "point-in-time panel and target definition, then tests a fixed-parameter LightGBM",
        "regressor on validation and internal-holdout splits. No 2025+ rows are loaded.",
        "",
        out.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Interpretation: LightGBM is used only as a data sanity check for nonlinear signal.",
        "The final strategy remains the transparent expanding-window linear score unless",
        "explicitly changed in a separate promotion process.",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    out = run(args.root.resolve())
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

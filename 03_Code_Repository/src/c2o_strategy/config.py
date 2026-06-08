"""Configuration for the C2O coursework pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StrategyConfig:
    """Run-time assumptions for the full strategy reproduction."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    start_date: str = "2010-01-01"
    cutoff: str = "2024-12-31"
    development_cutoff: str = "2024-12-31"
    universe_size: int = 1_000
    min_history_days: int = 252
    min_price: float = 5.0
    min_adv20: float = 10_000_000.0
    vol_floor: float = 0.05
    vol_cap: float = 1.20
    earnings_window_days: int = 1
    participation_cap: float = 0.05
    basket_fraction: float = 0.03
    min_basket_size: int = 15
    training_years: int = 4
    aums: tuple[float, ...] = field(
        default_factory=lambda: (50_000_000.0, 250_000_000.0, 1_000_000_000.0)
    )
    random_seed: int = 42

    @property
    def start_ts(self):
        import pandas as pd

        return pd.Timestamp(self.start_date)

    @property
    def cutoff_ts(self):
        import pandas as pd

        return pd.Timestamp(self.cutoff)

    @property
    def development_cutoff_ts(self):
        import pandas as pd

        return pd.Timestamp(self.development_cutoff)

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def model_cache_dir(self) -> Path:
        return self.output_dir.parent / "model_cache"

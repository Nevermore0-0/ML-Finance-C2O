import numpy as np
import pandas as pd

from c2o_strategy.metrics import max_drawdown, sharpe_ratio
from c2o_strategy.portfolio import capped_equal_allocation


def test_capped_equal_allocation_redistributes_until_caps_bind():
    caps = np.array([10.0, 100.0, 100.0])
    allocation = capped_equal_allocation(caps, 150.0)
    assert np.allclose(allocation, [10.0, 70.0, 70.0])
    assert allocation.sum() == 150.0


def test_capped_equal_allocation_reduces_when_capacity_insufficient():
    caps = np.array([10.0, 20.0])
    allocation = capped_equal_allocation(caps, 100.0)
    assert np.allclose(allocation, [10.0, 20.0])
    assert allocation.sum() == 30.0


def test_max_drawdown_uses_compounded_wealth():
    returns = pd.Series([0.10, -0.20, 0.05])
    assert np.isclose(max_drawdown(returns), -0.20)


def test_sharpe_ratio_returns_nan_for_flat_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(returns))

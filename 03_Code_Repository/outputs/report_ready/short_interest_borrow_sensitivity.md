# Short-Interest Borrow Sensitivity

The final promoted strategy uses tiered borrow costs only and does not
exclude high-short-interest names from selection. This diagnostic keeps
the same alpha score, top/bottom 3% basket rule, equal weighting, 5%
ADV20 cap, close-to-open execution, commission, slippage, and tiered
borrow charges, then asks what changes if the short leg excludes names
with decision-lagged DSI above 10%.

## Contribution Of DSI > 10% Shorts

| AUM   | Short position-days with DSI > 10%   |   Total gross-return contribution (bps) |   Average daily gross-return contribution (bps) | Share of all gross PnL   |
|:------|:-------------------------------------|----------------------------------------:|------------------------------------------------:|:-------------------------|
| 50M   | 12.97%                               |                                1289.54  |                                           0.342 | 4.84%                    |
| 250M  | 12.97%                               |                                1683.25  |                                           0.446 | 7.25%                    |
| 1B    | 12.97%                               |                                 579.371 |                                           0.154 | 6.16%                    |

## Hard-Exclusion Sensitivity

| AUM   |   Final Sharpe |   Hard-exclusion Sharpe | Final return   | Hard-exclusion return   | Final max DD   | Hard-exclusion max DD   | Final gross exposure   | Hard-exclusion gross exposure   |
|:------|---------------:|------------------------:|:---------------|:------------------------|:---------------|:------------------------|:-----------------------|:--------------------------------|
| 50M   |          1.308 |                   1.149 | 6.66%          | 5.72%                   | -13.82%        | -14.33%                 | 99.92%                 | 99.91%                          |
| 250M  |          1.445 |                   1.24  | 7.31%          | 5.98%                   | -10.19%        | -10.43%                 | 74.97%                 | 75.71%                          |
| 1B    |          1.293 |                   1.125 | 3.50%          | 2.89%                   | -5.56%         | -5.63%                  | 25.35%                 | 25.31%                          |

This sensitivity is not promoted as a new strategy. It is included to
answer the brief's borrow-honesty question and to show that the final
reported results are not produced by relaxing borrow or execution costs.

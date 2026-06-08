# Square-Root Impact Examples At 5% ADV20

These examples do not alter realised performance. The submitted
backtest keeps the fixed brief cost schedule of 0.5 bps commission per
leg and 1.5 bps auction slippage per leg. The calculation below is a
diagnostic reconciliation using `0.7 * sigma_daily * sqrt(0.05)`.

| example           | date       | ticker   |   instrument_id | market_cap   | adv20_dollar   | vol20_annual   | sigma_daily   | participation_cap   |   sqrt_impact_coefficient |   impact_at_5pct_adv_bps |
|:------------------|:-----------|:---------|----------------:|:-------------|:---------------|:---------------|:--------------|:--------------------|--------------------------:|-------------------------:|
| typical_mid_cap   | 2024-12-31 | EXE      |            1179 | $23,072.7M   | $220.6M        | 18.69%         | 1.18%         | 5.00%               |                       0.7 |                    18.43 |
| typical_large_cap | 2024-12-31 | FISV     |             215 | $115,856.9M  | $492.7M        | 9.79%          | 0.62%         | 5.00%               |                       0.7 |                     9.65 |

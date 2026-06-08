# Borrow Proxy Methodology

The final strategy uses a deterministic hard-to-borrow proxy based on lagged
short-interest stress variables already present in the point-in-time panel.

## Tier Rules

- Tier A: default, annual borrow rate 40 bps.
- Tier B: `dsi_lag1 >= 0.08`, or `dtcn_lag1 >= 5.0`, or `ddtcn_lag1 >= 1.0`; annual borrow rate 200 bps.
- Tier C: `dsi_lag1 >= 0.15`, or `dtcn_lag1 >= 10.0`, or both `dsi_lag1 >= 0.10` and `ddtcn_lag1 >= 1.5`; annual borrow rate 800 bps.

Borrow is charged daily on short notional as `borrow_rate_annual / 252`.

## Treatment In Final Strategy

No hard exclusion or short-ranking penalty is applied in the final promoted
strategy. The promoted Phase 2 challenger keeps the tiered-borrow-cost-only
short treatment exactly: short positions are selected from the low alpha-score
tail, and borrow affects reported net returns through explicit daily financing
cost.

The fraction of selected short names in Tier B/C is reported in
`outputs/report_ready/short_signal_affected_by_borrow.csv`. Because no hard
exclusion is implemented in the final strategy, the fraction of raw short signal
changed by borrow treatment is `0.0`; the fraction exposed to non-A-tier borrow
cost is separately reported.

## Tier Summary

|               AUM | borrow_tier   |   short_position_day_count |   share_of_short_position_days |   average_short_notional |   total_borrow_cost |   average_daily_borrow_bps |
|------------------:|:--------------|---------------------------:|-------------------------------:|-------------------------:|--------------------:|---------------------------:|
|   50000000.000000 | A             |                      40629 |                       0.432063 |           1084526.977560 |       699416.612242 |                   0.451248 |
|   50000000.000000 | B             |                      35972 |                       0.382538 |            948422.864661 |      2707672.006952 |                   0.451248 |
|   50000000.000000 | C             |                      17434 |                       0.185399 |            922914.403518 |      5107964.987595 |                   0.451248 |
|  250000000.000000 | A             |                      40629 |                       0.432063 |           4352833.192034 |      2807162.853320 |                   0.300775 |
|  250000000.000000 | B             |                      35972 |                       0.382538 |           3568609.022803 |     10188095.537163 |                   0.300775 |
|  250000000.000000 | C             |                      17434 |                       0.185399 |           2779389.714057 |     15382819.134877 |                   0.300775 |
| 1000000000.000000 | A             |                      40629 |                       0.432063 |           6247290.546136 |      4028907.422206 |                   0.097627 |
| 1000000000.000000 | B             |                      35972 |                       0.382538 |           4488488.009728 |     12814277.038567 |                   0.097627 |
| 1000000000.000000 | C             |                      17434 |                       0.185399 |           3613871.915819 |     20001346.977903 |                   0.097627 |

## Borrow-treatment Effect On Raw Short Signal

|               AUM |   raw_short_position_days |   tier_b_or_c_short_position_days |   fraction_tier_b_or_c |   selection_affected_by_borrow_treatment_days |   fraction_raw_short_signal_affected_by_borrow_treatment | hard_exclusion_sensitivity_implemented   | note                                                                                                                                         |
|------------------:|--------------------------:|----------------------------------:|-----------------------:|----------------------------------------------:|---------------------------------------------------------:|:-----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
|   50000000.000000 |                     94035 |                             53406 |               0.567937 |                                             0 |                                                 0.000000 | False                                    | Final strategy uses tiered borrow costs only. Borrow does not alter raw short selection; B/C names are affected in net returns through cost. |
|  250000000.000000 |                     94035 |                             53406 |               0.567937 |                                             0 |                                                 0.000000 | False                                    | Final strategy uses tiered borrow costs only. Borrow does not alter raw short selection; B/C names are affected in net returns through cost. |
| 1000000000.000000 |                     94035 |                             53406 |               0.567937 |                                             0 |                                                 0.000000 | False                                    | Final strategy uses tiered borrow costs only. Borrow does not alter raw short selection; B/C names are affected in net returns through cost. |

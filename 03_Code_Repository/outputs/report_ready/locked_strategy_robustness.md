# Locked Strategy Robustness

This diagnostic uses only the current champion strategy. No new strategy
variants are introduced here, and the current final outputs are not overwritten.
The 2025-2026 data remains outside the development window and should be treated
as held out for marker evaluation.

## Full-Period AUM Summary

|               AUM |   annual_return |   annual_vol |   sharpe |   max_drawdown |   average_gross_exposure_used |   fraction_cap_binding_days |
|------------------:|----------------:|-------------:|---------:|---------------:|------------------------------:|----------------------------:|
|   50000000.000000 |        0.066594 |     0.050256 | 1.308117 |      -0.138187 |                      0.999152 |                    0.939852 |
|  250000000.000000 |        0.073099 |     0.049673 | 1.445318 |      -0.101865 |                      0.749713 |                    0.999735 |
| 1000000000.000000 |        0.035015 |     0.026897 | 1.293031 |      -0.055599 |                      0.253463 |                    0.999735 |

## Pre/Post Split

|               AUM | subperiod       |   annual_return |   annual_vol |   sharpe |   max_drawdown |   average_gross_exposure_used |   fraction_cap_binding_days |
|------------------:|:----------------|----------------:|-------------:|---------:|---------------:|------------------------------:|----------------------------:|
|   50000000.000000 | split_2010_2016 |        0.071746 |     0.044690 | 1.572984 |      -0.138187 |                      0.998751 |                    1.000000 |
|   50000000.000000 | split_2017_2024 |        0.062102 |     0.054676 | 1.129381 |      -0.066422 |                      0.999503 |                    0.887177 |
|  250000000.000000 | split_2010_2016 |        0.044248 |     0.034492 | 1.272593 |      -0.101865 |                      0.604807 |                    1.000000 |
|  250000000.000000 | split_2017_2024 |        0.099020 |     0.059854 | 1.607691 |      -0.076432 |                      0.876614 |                    0.999503 |
| 1000000000.000000 | split_2010_2016 |        0.012655 |     0.009697 | 1.301778 |      -0.027112 |                      0.155653 |                    1.000000 |
| 1000000000.000000 | split_2017_2024 |        0.055001 |     0.035663 | 1.519303 |      -0.055599 |                      0.339121 |                    0.999503 |

## Stress Windows

|               AUM | subperiod            |   annual_return |   annual_vol |    sharpe |   max_drawdown |   average_gross_exposure_used |   fraction_cap_binding_days |
|------------------:|:---------------------|----------------:|-------------:|----------:|---------------:|------------------------------:|----------------------------:|
|   50000000.000000 | stress_late_2018     |       -0.064285 |     0.055391 | -1.172078 |      -0.026443 |                      1.000000 |                    1.000000 |
|   50000000.000000 | stress_2020_q1       |        0.249191 |     0.103221 |  2.207476 |      -0.044749 |                      1.000000 |                    0.983871 |
|   50000000.000000 | stress_2022_drawdown |        0.260484 |     0.070888 |  3.302410 |      -0.025297 |                      1.000000 |                    0.864542 |
|  250000000.000000 | stress_late_2018     |       -0.023513 |     0.047079 | -0.482211 |      -0.019200 |                      0.852784 |                    1.000000 |
|  250000000.000000 | stress_2020_q1       |        0.344139 |     0.107781 |  2.798751 |      -0.041378 |                      0.889505 |                    1.000000 |
|  250000000.000000 | stress_2022_drawdown |        0.322412 |     0.080238 |  3.524723 |      -0.026347 |                      0.992267 |                    1.000000 |
| 1000000000.000000 | stress_late_2018     |       -0.015311 |     0.014340 | -1.068924 |      -0.008075 |                      0.243824 |                    1.000000 |
| 1000000000.000000 | stress_2020_q1       |        0.060408 |     0.061953 |  0.977445 |      -0.022440 |                      0.308355 |                    1.000000 |
| 1000000000.000000 | stress_2022_drawdown |        0.195898 |     0.054942 |  3.284629 |      -0.023556 |                      0.484977 |                    1.000000 |

## Tail And Serial-Correlation Diagnostics

|               AUM |   worst_3m_return |   worst_6m_return |   worst_12m_return |   top_5pct_days_pnl_share |   return_autocorrelation_lag1 | sharpe_autocorr_note                    |
|------------------:|------------------:|------------------:|-------------------:|--------------------------:|------------------------------:|:----------------------------------------|
|   50000000.000000 |         -0.075107 |         -0.101770 |          -0.138053 |                  1.535409 |                     -0.013696 | no large positive lag-1 autocorrelation |
|  250000000.000000 |         -0.044618 |         -0.073142 |          -0.100903 |                  1.461259 |                     -0.006550 | no large positive lag-1 autocorrelation |
| 1000000000.000000 |         -0.028906 |         -0.044634 |          -0.049285 |                  1.727394 |                     -0.010668 | no large positive lag-1 autocorrelation |

The robustness evidence is mixed in the right way for a capacity-constrained
overnight strategy: 250M remains the cleanest reporting point, 1B is visibly
capacity constrained, and weak subperiods are not hidden. Positive lag-1
autocorrelation would make a naive annualised Sharpe look too optimistic; where
autocorrelation is small, that concern is less material.

# Stylised Fact Audit

This audit uses `outputs/equal_weight_return_decomposition.csv`, which is the
equal-weighted eligible-universe decomposition created from the daily panel.

| stream         |   cumulative_return |   annualised_return |   annualised_volatility |   sharpe |
|:---------------|--------------------:|--------------------:|------------------------:|---------:|
| overnight      |            2.978977 |            0.096600 |                0.117525 | 0.843923 |
| intraday       |            0.947889 |            0.045526 |                0.147759 | 0.375226 |
| close_to_close |            6.470237 |            0.143707 |                0.194294 | 0.788834 |

The overnight stream is stronger than the intraday stream in this equal-weight
large-cap universe, but the close-to-close stream is not mechanically identical
to an index-level S&P 500 decomposition. The report should describe the result
under this universe/equal-weight convention rather than claiming a perfect
replication of the index stylised fact.

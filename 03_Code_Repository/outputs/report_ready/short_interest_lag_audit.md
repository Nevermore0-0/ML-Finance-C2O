# Short-interest Lag Audit

The submitted pipeline uses the provided `all_data.parquet` short-interest proxies, including `dsi`, `dtcn`, and `ddtcn`, as already point-in-time transformed according to the coursework data package. The strategy then applies an additional one-trading-day decision lag in `add_point_in_time_features` before these fields enter features or borrow-tier assignment.

This audit verifies decision-time use of the provided point-in-time series. It
does not independently reconstruct raw FINRA settlement or publication dates
because those raw inputs are not present in the repository.

Hand-checkable examples are exported to `outputs/short_interest_lag_examples.csv`.

| ticker   |   instrument_id | source_feature_date   | decision_date_using_lag1   |   source_dsi |   source_dtcn |   source_ddtcn | rule_applied                                                                                                               | explanation                                                                                                                     |
|:---------|----------------:|:----------------------|:---------------------------|-------------:|--------------:|---------------:|:---------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| HLT      |               1 | 2015-01-27            | 2015-01-28                 |     0.004074 |      1.669489 |      -0.245679 | Short-interest proxies are read from the daily all_data panel, then shifted one trading day in add_point_in_time_features. | The code assumes the daily source already embeds the official publication/vendor lag; the strategy adds a one-day decision lag. |
| HLT      |               1 | 2015-01-28            | 2015-01-29                 |     0.004074 |      1.669489 |      -0.245679 | Short-interest proxies are read from the daily all_data panel, then shifted one trading day in add_point_in_time_features. | The code assumes the daily source already embeds the official publication/vendor lag; the strategy adds a one-day decision lag. |
| HLT      |               1 | 2015-01-29            | 2015-01-30                 |     0.004074 |      1.669489 |      -0.245679 | Short-interest proxies are read from the daily all_data panel, then shifted one trading day in add_point_in_time_features. | The code assumes the daily source already embeds the official publication/vendor lag; the strategy adds a one-day decision lag. |

# Borrow Proxy External Anecdotal Checks

These checks are intentionally small. They do not replace direct prime-broker
borrow-fee, locate-rate, or securities-lending validation, which is not
available in the coursework data package.

| ticker   | window_start   | window_end   | external_check                                                                                                                                                                                             | external_source                                                                                                                 |   internal_rows |   mean_dsi_lag1 |   max_dsi_lag1 |   mean_dtcn_lag1 |   max_dtcn_lag1 |   tier_b_or_c_fraction |   tier_c_fraction | internal_proxy_read                                                   |
|:---------|:---------------|:-------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------:|---------------:|-----------------:|----------------:|-----------------------:|------------------:|:----------------------------------------------------------------------|
| GME      | 2020-12-01     | 2021-02-15   | SEC staff report: GME short interest was around 100% of public float through most of 2020, reached 109.26% on 2020-12-31, and borrow fees were around 25% in January 2021 after exceeding 100% in Q2 2020. | https://www.sec.gov/files/staff-report-equity-options-market-struction-conditions-early-2021.pdf                                |              51 |           0.938 |          1.035 |            5.538 |          10.134 |                      1 |                 1 | Proxy flags the anecdotal crowded-short window as high borrow stress. |
| CVNA     | 2023-05-01     | 2023-07-31   | Reuters reported a May 2023 Carvana short-squeeze episode: the stock rose as much as 55%, short positions were about $488m, and Ortex described short sellers adding buy pressure while covering.          | https://www.investing.com/news/stock-market-news/usedcar-retailer-carvanas-shares-soar-on-upbeat-secondquarter-forecast-3074273 |              63 |           0.247 |          0.259 |            2.051 |           5.727 |                      1 |                 1 | Proxy flags the anecdotal crowded-short window as high borrow stress. |

Interpretation: the deterministic DSI/DTCN/DDTCN proxy catches two public
crowded-short episodes that are present in the local panel. This is still only
an anecdotal validation layer; the final backtest remains conservative by
charging explicit A/B/C borrow rates and by disclosing the missing direct
borrow-fee validation.

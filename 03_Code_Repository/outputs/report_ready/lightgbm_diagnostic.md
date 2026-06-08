# LightGBM Diagnostic

This diagnostic is not part of the final submitted strategy. It uses the same ranked
point-in-time panel and target definition, then tests a fixed-parameter LightGBM
regressor on validation and internal-holdout splits. No 2025+ rows are loaded.

| model                                           | lightgbm_version   | label                      | train_start   | train_end   | test_start   | test_end   | n_train_rows   |   n_test_rows |   daily_ic_count |   IC_mean |   IC_tstat | notes                                                                |
|:------------------------------------------------|:-------------------|:---------------------------|:--------------|:------------|:-------------|:-----------|:---------------|--------------:|-----------------:|----------:|-----------:|:---------------------------------------------------------------------|
| LightGBM                                        | 4.6.0              | validation_2019_2022       | 2010-01-01    | 2018-12-31  | 2019-01-01   | 2022-12-31 | 1760879        |        948788 |             1008 |  0.029263 |   6.288212 | Diagnostic only; fixed parameters; no portfolio promotion or tuning. |
| LightGBM                                        | 4.6.0              | internal_holdout_2023_2024 | 2010-01-01    | 2022-12-31  | 2023-01-01   | 2024-12-31 | 2709667        |        484478 |              502 |  0.020354 |   3.160769 | Diagnostic only; fixed parameters; no portfolio promotion or tuning. |
| Final transparent expanding-window linear score |                    | validation_2019_2022       |               |             | 2019-01-01   | 2022-12-31 |                |        948788 |             1008 |  0.019862 |   3.336067 | Reference final score IC over the same test period.                  |
| Final transparent expanding-window linear score |                    | internal_holdout_2023_2024 |               |             | 2023-01-01   | 2024-12-31 |                |        484478 |              502 |  0.018928 |   2.420786 | Reference final score IC over the same test period.                  |

Interpretation: LightGBM is used only as a data sanity check for nonlinear signal.
The final strategy remains the transparent expanding-window linear score unless
explicitly changed in a separate promotion process.

# Lo-Adjusted Sharpe Diagnostic

Uses Newey-West-style autocorrelation weights through lag `5`:
`1 + 2 * sum((1 - k/(q+1)) * rho_k)`.

| AUM   |   Raw net Sharpe |   Lo-adjusted net Sharpe |   Variance multiplier |   Lag-1 autocorr |
|:------|-----------------:|-------------------------:|----------------------:|-----------------:|
| 50M   |            1.308 |                    1.3   |                 1.012 |           -0.014 |
| 250M  |            1.445 |                    1.423 |                 1.032 |           -0.007 |
| 1B    |            1.293 |                    1.254 |                 1.063 |           -0.011 |

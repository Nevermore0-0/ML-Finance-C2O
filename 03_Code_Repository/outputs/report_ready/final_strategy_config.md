# Final Strategy Configuration

Final strategy: `phase2_g5_05_expanding`.

## Target

- Target: `volatility_scaled_overnight_return`
- Definition: `overnight_next / (vol20 / sqrt(252))`
- `overnight_next`: `adj_open_(t+1) / adj_close_t - 1`
- `vol20`: trailing 20-day close-to-close volatility, annualised, shifted by one trading day
- Feature set: unchanged from the original baseline point-in-time feature set

## Training Window And Alpha Learning

- Training window: expanding
- For each scored year Y, training starts at `2010-01-01` and ends before year Y begins
- The year being scored is never included in its own training sample
- Alpha learning method: 50% prior / 50% learned transparent correlation weights
- No additional features, external data, or 2025+ observations are introduced

## Universe Rule

- Annual top-1000 US common-equity universe by market capitalisation
- Universe is frozen each year using the last trading day of the previous year
- Minimum history requirement: `252` price observations

## Eligibility Filters

- Minimum price: `$5.00`
- Minimum ADV20: `$10,000,000`
- Volatility band: `5%` to `120%` annualised
- Earnings exclusion window: `+/- 1` trading day(s)
- Required data: next overnight return, lagged price, ADV20, and vol20

## Basket Selection and Weighting

- Selection: keep the selected experiment exactly as logged
- Long basket: top `3%` of eligible names by alpha score
- Short basket: bottom `3%` of eligible names by alpha score
- Minimum basket size: `15` names per side
- Weighting rule: equal weighting within each side, subject to capacity caps
- Ranking rule: raw alpha score
- Short treatment: tiered borrow cost only; no hard borrow exclusion
- Dollar neutrality: long and short books are matched after capacity sizing

## Capacity

- Participation cap: `5.0%` of ADV20 per name
- If a basket cannot absorb target side notional under caps, gross exposure is reduced

## AUM Levels

- 50M
- 250M
- 1B

## Cost Schedule

- Commission: 0.5 bps per leg, 1.0 bps round trip
- Auction slippage: 1.5 bps per leg, 3.0 bps round trip
- Total non-borrow round-trip cost: 4.0 bps on notional turnover

## Borrow Treatment

- Tier A: 40 bps per annum
- Tier B: 200 bps per annum
- Tier C: 800 bps per annum
- Borrow is charged daily on short notional as annual rate / 252
- No hard exclusion is applied in the final strategy; borrow affects net returns through cost

## Development Cutoff

- Start date: `2010-01-01`
- Development/data cutoff: `2024-12-31`
- No 2025+ data is used in development outputs

## Random Seed

- Configured seed: `42`
- The final strategy implementation is deterministic and does not use random sampling.

# Earnings Timing Audit

The final strategy uses `earnings_calendar.strat_trading_date` to respect the
provided BMO/AMC timing convention. Around each strategy trading date, the
eligibility filter excludes `+/- 1` trading day(s).

Hand-checkable examples are exported to `outputs/earnings_timing_examples.csv`.

| example_type         | ticker   |   instrument_id | reporting_date   | before_after_market   | raw_event_date   | strat_trading_date   | excluded_decision_date             | rule_applied                                                                              | explanation                                                                                          |
|:---------------------|:---------|----------------:|:-----------------|:----------------------|:-----------------|:---------------------|:-----------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| AMC earnings example | MOS      |              18 | 2010-01-05       | after                 | 2010-01-05       | 2010-01-05           | 2010-01-04, 2010-01-05, 2010-01-06 | After-market event is not known before the close; use provided strategy trading date.     | AMC earnings example: event is excluded around the strategy trading date using +/- 1 trading day(s). |
| BMO earnings example | RPM      |            1275 | 2010-01-06       | before                | 2010-01-06       | 2010-01-05           | 2010-01-04, 2010-01-05, 2010-01-06 | Before-market event is known before that day's close; use provided strategy trading date. | BMO earnings example: event is excluded around the strategy trading date using +/- 1 trading day(s). |
| Same-day example     | MOS      |              18 | 2010-01-05       | after                 | 2010-01-05       | 2010-01-05           | 2010-01-04, 2010-01-05, 2010-01-06 | After-market event is not known before the close; use provided strategy trading date.     | Same-day example: event is excluded around the strategy trading date using +/- 1 trading day(s).     |

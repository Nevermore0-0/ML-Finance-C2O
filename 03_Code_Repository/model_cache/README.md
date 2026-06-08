# Model Cache

This folder stores trained Phase 2 yearly feature-weight schedules.
The notebook still rebuilds the point-in-time panel from raw `data/`,
then applies these cached weights to score each stock-day and
regenerates positions, returns, report tables, figures, QuantStats,
and tests. If a compatible cache file is missing, stale, or uses a
different feature set/development cutoff, the notebook trains the
weights again and writes a fresh cache.

These files are model artefacts, not final performance CSVs.

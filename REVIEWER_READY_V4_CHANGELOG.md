# Reviewer-ready v4 changelog

Version 4 is an in-place update of the reviewer-ready v3 code bundle.

## Complete route baselines

- Added P0 Historical to the same mission-level comparison as P1 Shortest,
  P2 Minimum-Energy, P3 Maximum-EE, and P4 Multi-objective.
- P0 predicted Energy and EE use the same frozen segment model, feature space,
  and segment aggregation as P1-P4.
- Added per-policy distance change, predicted Energy saving, and predicted EE
  gain relative to predicted Historical.
- P3 Maximum-EE remains the proposed method. P4 Multi-objective remains a
  supporting operational/sensitivity strategy.

## Canonical new results

- `output/results/multihub/route_baseline_comparison.csv`
- `output/results/multihub/route_baseline_summary.csv`
- `output/results/multihub/observed_route_baseline_validation.csv`

The summary has exactly five rows: Historical, Shortest-distance,
Minimum-energy, Maximum-EE, and Multi-objective.

## Fairness separation

- `route_baseline_comparison.csv` is the primary model-to-model comparison.
- `observed_route_baseline_validation.csv` is a supporting realism check and
  explicitly labels observed Historical versus predicted candidates.
- Existing final RQ5 fair and observed tables remain because they support a
  different, recommendation-specific claim.

## Removed superseded results

- `historical_strategy_comparison.csv`
- `historical_strategy_summary.csv`

These files are deleted only after their canonical P0-P4 replacements have
been written successfully.

## Verification

- Route-baseline comparison: 15 missions, 55 columns.
- Route-baseline summary: 5 strategies, 13 columns.
- Observed supporting comparison: 15 missions, 31 columns.
- Multi-objective weight sensitivity: 45 mission-weight evaluations.
- Frozen ML model retained; no model retraining or feature change.

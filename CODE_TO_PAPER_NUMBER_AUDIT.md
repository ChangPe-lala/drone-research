# Code-to-paper number audit

Audit date: 2026-08-26  
Scope: current code and generated outputs in this workspace. No models or result files were regenerated during this audit.

## Executive status

The principal numerical claims in the checklist are reproducible from the current outputs, with two important qualifications:

1. `3.62%`, `3.880%`, and `4.113%` are all valid but use different populations/decision treatments and must be labelled explicitly.
2. The current paper identifies the combined-domain model as LightGBM, while the current model manifest and holdout table identify it as ElasticNet. This is a paper/code mismatch that must be resolved before submission.

The `96.3%` statistic is reproducible only on the 63-row combined holdout. On all 262 harmonized rows the corresponding share is `89.401%`. The paper must state the population.

## Checklist audit

### 1. Table III aggregation — PASS, paper clarification required

For each route in each mission:

- route energy is the sum of predicted segment Wh;
- route EE is `route_distance / route_energy`;
- Table III distance, energy and EE are separate arithmetic means across the 15 missions;
- delta energy and delta EE are means of paired, mission-level percentage changes.

Therefore, the displayed `mean(EE_i)` is not expected to equal `mean(D_i) / mean(E_i)`.

Example for P0 Historical:

- mean distance = `598.850900 m`;
- mean predicted energy = `21.294492 Wh`;
- mean mission-level EE = `28.714217 m/Wh`;
- ratio of displayed means = `598.850900 / 21.294492 = 28.122338 m/Wh`.

The difference is an aggregation effect, not a formula error.

Code trace:

- `src/multihub/enumerate_candidate_paths.py`: sums edge predictions and calculates per-route EE.
- `src/multihub/strategy_comparison.py`: calculates paired mission-level changes, then averages each metric column.
- `output/results/multihub/route_baseline_comparison.csv`: mission-level source table.
- `output/results/multihub/route_baseline_summary.csv`: five-row Table III source.

Recommended Table III note:

> D, predicted energy, and EE are arithmetic means of mission-level values. Mean EE is mean(D_i/E_i), not mean(D_i)/mean(E_i). Delta values are means of paired mission-level percentage changes. Positive energy change denotes a reduction; negative values denote increased consumption.

### 2. `3.62%` versus `3.880%` versus `4.113%` — PASS, labels required

- `3.621658%` (reported as `3.62%`): mean final-decision EE gain across all 15 missions. Fourteen missions use P3; the rejected mission retains P0 and contributes `0%` gain.
- `3.880348%`: mean candidate EE gain over the 14 accepted P3 alternatives only.
- `4.112705%`: median candidate EE gain over the same 14 accepted P3 alternatives.
- Related but different Table III value: `3.545094%` is the mean P3-versus-P0 gain over all 15 raw P3 candidates, including the candidate that was ultimately rejected.

These values are internally coherent. The abstract's `3.62%` describes the conservative final policy on 15 missions; Results must explicitly say that `3.880%` and `4.113%` are conditional on the 14 accepted alternatives.

### 3. Pooled split `199/63` — PASS

The combined split is constructed independently. For COMBINED, the code samples holdout date groups separately inside DJI and VTOL and then concatenates the resulting masks. It does not combine the independently generated DJI-only and VTOL-only partitions.

Current manifests:

- DJI_ONLY: `146/48` from 194 rows;
- VTOL_ONLY: `41/27` from 68 rows;
- COMBINED: `199/63` from 262 rows.

Thus `199/63` need not equal `146+41 / 48+27`.

Code trace: `src/modeling/validation_common.py`, functions `_platform_holdout_groups` and `make_untouched_holdout`.

### 4. Platform-mean separation `96.3%` — PASS only for the holdout population

Using the 63-row combined holdout:

```text
grand_mean = mean(y)
platform_mean_i = mean(y | platform_i)
SS_between = sum_i (platform_mean_i - grand_mean)^2
SS_total   = sum_i (y_i - grand_mean)^2
share_pct  = 100 * SS_between / SS_total
```

Recalculated result:

- holdout counts: DJI `48`, VTOL `15`;
- holdout means: DJI `28.971098`, VTOL `98.815376 m/Wh`;
- `100 * SS_between / SS_total = 96.298976%`.

On all 262 harmonized rows, the same formula gives only `89.400948%`. The paper must say “in the combined holdout” and should not imply this was calculated on all data.

No dedicated production script currently writes this statistic to an audit artifact. It should be added for reproducibility rather than remaining a manual calculation.

### 5. Segment-model `R2 = 0.9775 +/- 0.0070` — PASS

This is not an independent test result. It is the arithmetic mean and standard deviation of five fold-level R2 values from `GroupKFold`, grouped by `flight_uid`:

- fold R2: `0.986479`, `0.967278`, `0.982729`, `0.972137`, `0.978755`;
- mean: `0.977475508`;
- standard deviation: `0.006968401`;
- pooled OOF R2 is separately available and equals `0.977888218`.

The correct wording is “mean 5-fold flight-grouped cross-validated R2,” not independent-test R2.

Code/output trace:

- `src/multihub/segment_modeling.py`;
- `output/results/multihub/segment_model_checkpoint/best_config_fold_metrics.csv`;
- `output/results/multihub/segment_model_checkpoint/segment_model_evidence.csv`.

### 6. Why P1 equals P2 — PASS

The policy code correctly defines:

- P1 = `argmin(total_distance)`;
- P2 = `argmin(predicted_total_energy)`;
- both operate only on eligible hypothetical alternatives; P0 historical is excluded.

Across the 15 historical requests there are 86 eligible hypothetical paths. P1 and P2 select the same route in all 15 missions.

Diagnostic evidence:

- overall Pearson correlation between candidate distance and predicted energy: `0.608178`;
- overall Spearman correlation: `0.645414`;
- mean within-mission Pearson correlation: `0.826935`;
- mean within-mission Spearman correlation: `0.964699`.

Thus the equality is caused by strong request-local monotonic ordering between distance and predicted Wh in the available alternatives. It does not mean the two objectives are mathematically identical in general.

### 7. Are hypothetical routes unseen? — PASS as bounded recombination, limitation required

The graph is constructed only from directed edges supported by historical segments. Candidate paths are simple paths formed by recombining these historically supported local edges under platform/dataset, hub, hop-count, and detour constraints.

Consequently, the experiment does not establish generalization to entirely unseen geographic corridors. Recommended limitation:

> The routing experiment evaluates recombinations of historically observed local trajectory segments rather than entirely unseen geographic corridors.

### 8. Battery-preservation interpretation — PASS

P3 should not be selected if absolute battery preservation is the primary objective. In Table III, P3 improves mean predicted EE by `3.545094%` but its mean energy-saving metric is `-4.234073%`, meaning predicted energy consumption increases. P2 is the appropriate policy for minimum predicted Wh; P3 is a higher-EE screening policy.

## Five required numerical traces

| Claim | Exact current value | Population/protocol | Primary evidence |
|---|---:|---|---|
| DJI R2 | `0.929937114` | 48-row untouched DJI holdout | `validation/dji_only/finalized/holdout_metrics.csv` |
| Combined R2 | `0.974822878` | 63-row independently constructed combined holdout | `validation/combined/finalized/holdout_metrics.csv` |
| Platform separation | `96.298976%` | same 63-row combined holdout | recomputed from combined `holdout_predictions.csv` |
| Segment CV R2 | `0.977475508 +/- 0.006968401` | 5-fold GroupKFold by 194 flights, 590 segments | segment checkpoint evidence |
| Accepted P3 gain | mean `3.880348%`, median `4.112705%` | 14 accepted alternatives | final RQ5 fair comparison |
| Hub-removal stability | `75.735294%` | 103 unchanged routes among 136 comparable topology selections | dominant-hub removal summary |

## Additional critical mismatch found

The current combined-domain finalized manifest states:

- model: `ElasticNet`;
- parameters: `alpha=0.1`, `l1_ratio=0.8`;
- holdout R2: `0.974822878`.

The current paper reports the same R2 but labels the combined model as LightGBM. The model name in the paper must be changed to ElasticNet unless a different, archived run can prove that `0.974822878` came from LightGBM. Using one run's metric with another model's label is not acceptable.

## Reproducibility gaps to fix in code

1. Add a dedicated, versioned audit script/output for platform-mean separation and all `3.62/3.880/4.113/3.545` populations.
2. Add an automated consistency assertion that every selected P1-P4 path equals the corresponding argmin/argmax recomputed from `candidate_paths.csv`.
3. Add the within-request distance-energy correlations to a reviewer-facing CSV.
4. Update `check_stats.py`: it looks for the obsolete path `output/results/multihub/segment_model_evidence.csv`; the current evidence lives under `segment_model_checkpoint/`.
5. Record hashes or run identifiers in summary artifacts so tables from different runs cannot be silently mixed.

## Test-suite status

`python -m pytest -q` did not reach test execution because collection failed:

- the active `.venv` does not contain `networkx`;
- `tests/test_battery_scheduler.py` and `tests/test_routing_policies.py` import the obsolete/nonexistent `experiments` package.

This does not invalidate the CSV recomputations above, but the repository cannot currently claim a clean automated test run. Dependencies and stale test imports should be repaired, followed by a full rerun.

## Submission conclusion

The routing calculations and the five requested numerical traces are reproducible from the current output set. The paper is not yet numerically ready until the combined-model name mismatch is corrected and the populations behind `96.3%` and the four EE-gain values are explicitly stated.

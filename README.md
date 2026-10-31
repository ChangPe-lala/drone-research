# DTA ML Pipeline (Reproducible Execution)

This repository contains the machine learning pipeline for DTA. It is designed around a strict hybrid execution approach balancing R&D flexibility with absolute reproducibility for official reporting.


## How to run the official experiment

```bash
# 1. Start the pipeline
python run_pipeline.py --stage all

# 2. The pipeline will stop at GATE 1 (Data Quality). 
# Review output/results/data_quality/outlier_candidates.csv
# Update decisions/data_quality_decision.yaml (set approved: true)

# 3. Resume the pipeline
python run_pipeline.py --stage all

# 4. The pipeline will stop at GATE 2 (Split Protocol).
# Review output/figures/splits/
# Update decisions/split_decision.yaml (set approved: true)

# 5. Resume the pipeline
python run_pipeline.py --stage all

# 6. The pipeline will stop at GATE 3 (Model Selection).
# Review output/results/model_evaluation/model_evidence_table.csv
# Update decisions/model_selection_decision.yaml (set approved: true)

# 7. Resume the pipeline to evaluate the final test set and lock the experiment.
python run_pipeline.py --stage all
```

## Directory Structure
- `config/`: System paths and feature definitions.
- `decisions/`: YAML files capturing human decisions (Data Quality, Split, Model Selection).
- `data/`: Raw, intermediate, and processed CSV files (not tracked in git).
- `src/`: Pipeline source code (`data_processing/`, `modeling/`, `analysis/`, `utils/`).
- `output/`: Generated data, figures, models, results, and metadata tracking (Hashes, Manifests).
- `tests/`: Unit tests and leakage checks.
- `docs/`: Methodology and protocols.


## Multi-UAV EE patch — August 2026

The current primary target is `energy_efficiency`. Shared target integration now uses the same `0 < dt <= 30 s` segment-validity rule for distance and battery energy. Observed `relative_wind_angle` is analysis-only. Route-weather aerodynamic features are supported, but historical Start→End geometry is explicitly tagged as `OBSERVED_ENDPOINT_PROXY`; final pre-flight use requires true planned route/edge geometry. See `CODE_AUDIT_AND_PATCH_NOTES.md`.

Recommended checkpoint: run `python run_dataset_extension.py --stage D06A`, review the target audit, then rebuild D03→D06 before final D07/D08.

## Reviewer-ready Multi-Hub routing experiment

The final routing run now separates two inference layers:

- `HISTORICAL_MISSION`: mission-specific requests used for the primary fair
  predicted-vs-predicted RQ5 comparison and the supporting observed baseline.
- `TOPOLOGY_OD`: one representative operating-condition scenario for every
  ordered, same-platform OD with at least two connected, plausible Multi-Hub
  paths. These scenarios test graph-wide routing generality and are not counted
  as independent observed missions.

Run the canonical baseline after freezing the segment model:

```bash
python run_final_multihub.py --case DJI_ONLY
```

Primary outputs are:

- `output/data/multihub/routing_experiment_scope.csv`
- `output/results/multihub/final_rq5/fair_model_based_routing_comparison.csv`
- `output/results/multihub/final_rq5/observed_baseline_validation.csv`
- `output/results/multihub/topology_audit/topology_node_audit.csv`
- `output/results/multihub/topology_audit/dominant_hub_removal_summary.csv`

The same historical missions are also evaluated under four route-selection
strategies:

- P1 Shortest Distance: minimize route distance.
- P2 Energy-first: minimize predicted absolute Energy (Wh).
- P3 EE-first: maximize predicted distance/Energy.
- P4 Multi-objective: balance normalized EE benefit and inverse normalized
  Energy cost using
  `Score = w_EE * EE_norm + w_Energy * (1 - Energy_norm)`.

P4 uses 0.50/0.50 for the main comparison. Its weight sensitivity is reported
at 0.25/0.75, 0.50/0.50, and 0.75/0.25. New reviewer-facing outputs are:

- `output/results/multihub/route_baseline_comparison.csv`
- `output/results/multihub/route_baseline_summary.csv`
- `output/results/multihub/observed_route_baseline_validation.csv`
- `output/results/multihub/multi_objective_weight_sensitivity.csv`
- `output/results/multihub/multi_objective_weight_sensitivity_summary.csv`
- `output/data/multihub/multi_objective_candidate_scores.csv`

The canonical route-baseline comparison includes the recorded Historical route
(P0) alongside P1-P4. P0 and P1-P4 predicted metrics use the same frozen
segment model. The separate observed-baseline table is supporting only because
it compares actual Historical measurements with predicted candidate routes.
The superseded v3 files `historical_strategy_comparison.csv` and
`historical_strategy_summary.csv` are removed when the new tables are written.

Run the five one-factor-at-a-time graph configurations with the same frozen ML
model:

```bash
python run_multihub_sensitivity.py --case DJI_ONLY
```

The sensitivity runner isolates each configuration and reports topology, hub,
and recommendation stability under `output/results/multihub/sensitivity/`.

# DTA ML Pipeline (Reproducible Execution)

This repository contains the machine learning pipeline for DTA. It is designed around a strict hybrid execution approach balancing R&D flexibility with absolute reproducibility for official reporting.



## Canonical Research Pipeline

### 1. Data preparation and supporting experiments

python run_dataset_extension.py --stage all

Runs D00-D12, including D06A and D07A.
Does not automatically finalize the research configuration.

### 2. CP1 — Feature screening

python run_pipeline.py --stage cp1 --case DJI_ONLY --models all

Review feature_evidence.csv before selecting a feature set.
Repeat for VTOL_ONLY and COMBINED.

### 3. CP2 — Model benchmark

python run_pipeline.py --stage cp2 --case DJI_ONLY --feature-set FS_OPS_ENV --models all

FS_OPS_ENV is an example; select the feature set using CP1 evidence.

### 4. Finalize — Holdout, SHAP, Ablation

python run_pipeline.py --stage finalize --case DJI_ONLY --feature-set FS_OPS_ENV --model CatBoost

The feature set and model must be selected explicitly using research evidence.
Repeat for other validation cases as required.

### 5. CP3 — Compare and select

python run_pipeline.py --stage cp3
python run_pipeline.py --stage select --case DJI_ONLY

Select the final case only after reviewing CP3 evidence.

### 6. Full Ablation

python run_full_ablation.py

### 7. Multi-Hub G00-G07

python run_multihub_g00_g07.py --case DJI_ONLY --models all

### 8. Freeze segment Energy model

python fit_segment_energy_model.py --model XGBoost

Choose the segment model using benchmark evidence.

### 9. Multi-Hub G08-G14 and P0-P4

python run_final_multihub.py --case DJI_ONLY

### 10. Topology sensitivity

python run_multihub_sensitivity.py --case DJI_ONLY

### 11. Check pipeline status

python run_research.py status
python run_pipeline.py --stage status

Research decisions are explicit.
No automatic CP1/CP2/CP3 approval or selection is performed.


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

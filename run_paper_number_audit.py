"""Reproduce the reviewer-facing numerical claims from finalized outputs.

This script is intentionally read-only with respect to model/routing artifacts.
It reads finalized outputs, recomputes the paper statistics, performs structural
consistency checks, and writes reviewer-facing audit artifacts under
``output/results/paper_audit``.

Run from the repository root:

    python run_paper_number_audit.py

The audit must be run only after the platform cases, segment model, final
Multi-Hub experiment, and topology audit have completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "results" / "paper_audit"

COMBINED_FEATURES = (
    PROJECT_ROOT
    / "output"
    / "data"
    / "features"
    / "multiuav_flight_level_features.csv"
)
VALIDATION_DIR = (
    PROJECT_ROOT / "output" / "results" / "multiuav" / "validation"
)
COMBINED_HOLDOUT = (
    VALIDATION_DIR / "combined" / "finalized" / "holdout_predictions.csv"
)
ROUTE_COMPARISON = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "multihub"
    / "route_baseline_comparison.csv"
)
FAIR_COMPARISON = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "multihub"
    / "final_rq5"
    / "fair_model_based_routing_comparison.csv"
)
SEGMENT_EVIDENCE = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "multihub"
    / "segment_model_checkpoint"
    / "segment_model_evidence.csv"
)
SELECTED_SEGMENT_MODEL = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "multihub"
    / "segment_model_checkpoint"
    / "selected_segment_model.json"
)
TOPOLOGY_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "results"
    / "multihub"
    / "topology_audit"
    / "dominant_hub_removal_summary.csv"
)


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required audit input is missing: {path}\n"
            "Run the corresponding finalized pipeline stage first."
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with _require_file(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _finite(value: Any, label: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Audit value '{label}' is not finite: {value}")
    return numeric


def _manifest(case: str) -> dict[str, Any]:
    path = VALIDATION_DIR / case / "finalized" / "selection_manifest.json"
    return _read_json(path)


def audit_validation_cases() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for case in ("dji_only", "vtol_only", "combined"):
        manifest = _manifest(case)
        manifests[case] = manifest
        split = manifest["split"]
        metrics = manifest["holdout_metrics"]
        n_total = int(split["n_total"])
        n_development = int(split["n_development"])
        n_holdout = int(split["n_holdout"])
        if n_total != n_development + n_holdout:
            raise AssertionError(
                f"{case}: n_total != n_development + n_holdout"
            )
        if int(metrics["N"]) != n_holdout:
            raise AssertionError(
                f"{case}: holdout metric N does not match split holdout N"
            )
        rows.append(
            {
                "case": manifest["case"],
                "model": manifest["model"],
                "feature_set": manifest["feature_set"],
                "n_total": n_total,
                "n_development": n_development,
                "n_holdout": n_holdout,
                "holdout_r2": _finite(metrics["R2"], f"{case}.R2"),
                "holdout_mae": _finite(metrics["MAE"], f"{case}.MAE"),
                "holdout_rmse": _finite(metrics["RMSE"], f"{case}.RMSE"),
                "holdout_wmape": _finite(
                    metrics["WMAPE"], f"{case}.WMAPE"
                ),
                "holdout_date_groups_by_dataset": json.dumps(
                    split["holdout_date_groups_by_dataset"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows), manifests


def _platform_separation(
    df: pd.DataFrame,
    *,
    population: str,
    source_file: Path,
) -> dict[str, Any]:
    target = "energy_efficiency"
    platform = "dataset_id"
    missing = {target, platform} - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_file} is missing columns: {sorted(missing)}"
        )

    clean = df[[platform, target]].copy()
    clean[target] = pd.to_numeric(clean[target], errors="coerce")
    clean = clean[np.isfinite(clean[target])].copy()
    if clean.empty or clean[platform].nunique() < 2:
        raise ValueError(
            f"{population}: need finite targets from at least two platforms"
        )

    y = clean[target].to_numpy(float)
    grand_mean = float(np.mean(y))
    grouped = clean.groupby(platform, sort=True)[target]
    counts = grouped.size()
    means = grouped.mean()
    ss_platform = float(
        sum(
            int(counts.loc[name])
            * (float(means.loc[name]) - grand_mean) ** 2
            for name in means.index
        )
    )
    ss_total = float(np.sum((y - grand_mean) ** 2))
    if ss_total <= 0:
        raise ValueError(f"{population}: total target sum of squares is zero")

    row: dict[str, Any] = {
        "population": population,
        "source_file": str(source_file.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(clean)),
        "grand_mean_ee": grand_mean,
        "ss_platform": ss_platform,
        "ss_total": ss_total,
        "platform_separation_fraction": ss_platform / ss_total,
        "platform_separation_pct": 100.0 * ss_platform / ss_total,
        "formula": (
            "100 * sum_g[n_g*(mean_g-grand_mean)^2] / "
            "sum_i[(y_i-grand_mean)^2]"
        ),
    }
    for name in means.index:
        safe_name = str(name).lower().replace(" ", "_")
        row[f"n_{safe_name}"] = int(counts.loc[name])
        row[f"mean_ee_{safe_name}"] = float(means.loc[name])
    return row


def audit_platform_separation(
    manifests: dict[str, Any],
) -> pd.DataFrame:
    holdout = pd.read_csv(_require_file(COMBINED_HOLDOUT))
    full = pd.read_csv(_require_file(COMBINED_FEATURES))

    expected_holdout = int(manifests["combined"]["split"]["n_holdout"])
    if len(holdout) != expected_holdout:
        raise AssertionError(
            "Combined holdout_predictions row count does not match manifest"
        )
    if len(full) != int(manifests["combined"]["split"]["n_total"]):
        raise AssertionError(
            "Combined feature-table row count does not match manifest"
        )

    return pd.DataFrame(
        [
            _platform_separation(
                holdout,
                population="COMBINED_UNTOUCHED_HOLDOUT",
                source_file=COMBINED_HOLDOUT,
            ),
            _platform_separation(
                full,
                population="COMBINED_FULL_HARMONIZED_DATASET",
                source_file=COMBINED_FEATURES,
            ),
        ]
    )


def audit_routing_gains() -> pd.DataFrame:
    fair = pd.read_csv(_require_file(FAIR_COMPARISON))
    route = pd.read_csv(_require_file(ROUTE_COMPARISON))
    required_fair = {
        "predicted_historical_ee",
        "predicted_candidate_ee",
        "predicted_ee_gain_pct",
        "recommended_ee",
        "recommendation_decision",
    }
    missing = required_fair - set(fair.columns)
    if missing:
        raise ValueError(f"Fair comparison missing columns: {sorted(missing)}")
    if "max_ee_gain_vs_historical_pct" not in route.columns:
        raise ValueError(
            "Route comparison lacks max_ee_gain_vs_historical_pct"
        )

    expected_gain = (
        (fair["predicted_candidate_ee"] - fair["predicted_historical_ee"])
        / fair["predicted_historical_ee"]
        * 100.0
    )
    if not np.allclose(
        expected_gain,
        fair["predicted_ee_gain_pct"],
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        raise AssertionError(
            "Stored predicted_ee_gain_pct does not match recomputed gain"
        )

    final_gain = (
        (fair["recommended_ee"] - fair["predicted_historical_ee"])
        / fair["predicted_historical_ee"]
        * 100.0
    )
    accepted_mask = (
        fair["recommendation_decision"] == "USE_HYPOTHETICAL_ROUTE"
    )
    retained_mask = (
        fair["recommendation_decision"] == "RETAIN_HISTORICAL_ROUTE"
    )
    if not bool((accepted_mask | retained_mask).all()):
        raise AssertionError("Unknown recommendation decision found")
    if not np.allclose(final_gain.loc[retained_mask], 0.0, atol=1e-10):
        raise AssertionError(
            "A retained historical decision has non-zero final-policy gain"
        )

    accepted_gain = fair.loc[accepted_mask, "predicted_ee_gain_pct"]
    raw_p3_gain = route["max_ee_gain_vs_historical_pct"]
    return pd.DataFrame(
        [
            {
                "metric_id": "RAW_P3_MEAN_GAIN",
                "value_pct": float(raw_p3_gain.mean()),
                "aggregation": "mean",
                "population": "all raw P3 candidates before accept/reject",
                "n_population": int(raw_p3_gain.notna().sum()),
                "source_file": str(ROUTE_COMPARISON.relative_to(PROJECT_ROOT)),
            },
            {
                "metric_id": "ACCEPTED_P3_MEAN_GAIN",
                "value_pct": float(accepted_gain.mean()),
                "aggregation": "mean",
                "population": "accepted P3 hypothetical alternatives",
                "n_population": int(accepted_gain.notna().sum()),
                "source_file": str(FAIR_COMPARISON.relative_to(PROJECT_ROOT)),
            },
            {
                "metric_id": "ACCEPTED_P3_MEDIAN_GAIN",
                "value_pct": float(accepted_gain.median()),
                "aggregation": "median",
                "population": "accepted P3 hypothetical alternatives",
                "n_population": int(accepted_gain.notna().sum()),
                "source_file": str(FAIR_COMPARISON.relative_to(PROJECT_ROOT)),
            },
            {
                "metric_id": "FINAL_POLICY_MEAN_GAIN",
                "value_pct": float(final_gain.mean()),
                "aggregation": "mean",
                "population": (
                    "all final decisions; retained Historical contributes 0%"
                ),
                "n_population": int(final_gain.notna().sum()),
                "source_file": str(FAIR_COMPARISON.relative_to(PROJECT_ROOT)),
            },
        ]
    )


def audit_segment_model() -> pd.DataFrame:
    evidence = pd.read_csv(_require_file(SEGMENT_EVIDENCE))
    selected = _read_json(SELECTED_SEGMENT_MODEL)
    model_name = str(selected["model_name"])
    row = evidence[evidence["Model"] == model_name]
    if len(row) != 1:
        raise AssertionError(
            f"Expected one evidence row for selected segment model {model_name}"
        )
    result = row.copy()
    result.insert(0, "selected_for_routing", True)
    result.insert(
        1,
        "interpretation",
        "5-fold GroupKFold grouped by flight_uid; not independent holdout",
    )
    return result


def audit_topology() -> pd.DataFrame:
    summary = pd.read_csv(_require_file(TOPOLOGY_SUMMARY))
    if len(summary) != 1:
        raise AssertionError("Expected exactly one topology summary row")
    row = summary.iloc[0]
    n = int(row["n_comparable_recommendations"])
    rate = _finite(
        row["same_route_rate_after_removal"],
        "same_route_rate_after_removal",
    )
    unchanged = int(round(n * rate))
    if not np.isclose(unchanged / n, rate, atol=1e-12):
        raise AssertionError(
            "Topology same-route rate is inconsistent with an integer count"
        )
    return pd.DataFrame(
        [
            {
                "dominant_hub": row["dominant_hub"],
                "n_comparable_recommendations": n,
                "n_same_route_after_removal": unchanged,
                "same_route_rate_after_removal": rate,
                "same_route_pct_after_removal": 100.0 * rate,
                "routability_loss_rate": _finite(
                    row["routability_loss_rate"], "routability_loss_rate"
                ),
                "mean_best_ee_change_pct_after_removal": _finite(
                    row["mean_best_ee_change_pct_after_removal"],
                    "mean_best_ee_change_pct_after_removal",
                ),
                "evidence_pattern": row["evidence_pattern"],
                "source_file": str(TOPOLOGY_SUMMARY.relative_to(PROJECT_ROOT)),
            }
        ]
    )


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if df.empty:
        return "_No rows._"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.9g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _markdown_report(
    validation: pd.DataFrame,
    platform: pd.DataFrame,
    routing: pd.DataFrame,
    segment: pd.DataFrame,
    topology: pd.DataFrame,
) -> str:
    combined = validation[validation["case"] == "COMBINED"].iloc[0]
    holdout_sep = platform[
        platform["population"] == "COMBINED_UNTOUCHED_HOLDOUT"
    ].iloc[0]
    full_sep = platform[
        platform["population"] == "COMBINED_FULL_HARMONIZED_DATASET"
    ].iloc[0]
    routing_lookup = routing.set_index("metric_id")["value_pct"]
    seg = segment.iloc[0]
    topo = topology.iloc[0]
    return f"""# Paper-number audit

Generated from finalized repository outputs by `run_paper_number_audit.py`.

## Validation cases

{_markdown_table(validation)}

## Combined split and platform separation

- Combined model: **{combined['model']}**.
- Combined split: **{int(combined['n_development'])}/{int(combined['n_holdout'])}** development/holdout from {int(combined['n_total'])} flights.
- Holdout platform separation: **{holdout_sep['platform_separation_pct']:.6f}%** over {int(holdout_sep['n_rows'])} rows.
- Full harmonized-data separation: **{full_sep['platform_separation_pct']:.6f}%** over {int(full_sep['n_rows'])} rows.
- Formula: `{holdout_sep['formula']}`.

## Routing-gain populations

{_markdown_table(routing)}

Key distinction:

- Raw P3 mean gain over all candidates: **{routing_lookup['RAW_P3_MEAN_GAIN']:.6f}%**.
- Accepted P3 mean/median gain: **{routing_lookup['ACCEPTED_P3_MEAN_GAIN']:.6f}% / {routing_lookup['ACCEPTED_P3_MEDIAN_GAIN']:.6f}%**.
- Final-policy mean gain over all decisions: **{routing_lookup['FINAL_POLICY_MEAN_GAIN']:.6f}%**.

## Segment model

- Selected model: **{seg['Model']}**.
- Mean grouped-CV R²: **{seg['CV_R2_mean']:.6f} ± {seg['CV_R2_std']:.6f}**.
- Pooled OOF R²: **{seg['OOF_R2']:.6f}**.
- Mean grouped-CV WMAPE: **{100.0 * seg['CV_WMAPE_mean']:.4f}%**.
- Population: **{int(seg['N_Segments'])} segments / {int(seg['N_Flights'])} flights**.
- Interpretation: {seg['interpretation']}.

## Dominant-hub removal

- Dominant hub: **{topo['dominant_hub']}**.
- Same route after removal: **{int(topo['n_same_route_after_removal'])}/{int(topo['n_comparable_recommendations'])} = {topo['same_route_pct_after_removal']:.6f}%**.
- Routability loss: **{100.0 * topo['routability_loss_rate']:.6f}%**.
- Mean best-EE change: **{topo['mean_best_ee_change_pct_after_removal']:.6f}%**.

## Audit status

All structural consistency checks passed. These values describe the current finalized outputs and must be regenerated if data, splits, models, graph construction, or routing policies change.
"""


def run_audit(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    validation, manifests = audit_validation_cases()
    platform = audit_platform_separation(manifests)
    routing = audit_routing_gains()
    segment = audit_segment_model()
    topology = audit_topology()

    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "validation_case_metrics.csv", index=False)
    platform.to_csv(output_dir / "platform_mean_separation.csv", index=False)
    routing.to_csv(output_dir / "routing_gain_populations.csv", index=False)
    segment.to_csv(output_dir / "segment_model_metrics.csv", index=False)
    topology.to_csv(output_dir / "topology_stability.csv", index=False)

    payload = {
        "audit_version": 1,
        "project_root": str(PROJECT_ROOT),
        "status": "PASS",
        "validation_cases": _records(validation),
        "platform_mean_separation": _records(platform),
        "routing_gain_populations": _records(routing),
        "segment_model": _records(segment),
        "topology_stability": _records(topology),
    }
    with (output_dir / "paper_number_audit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    report = _markdown_report(
        validation,
        platform,
        routing,
        segment,
        topology,
    )
    (output_dir / "PAPER_NUMBER_AUDIT.md").write_text(
        report, encoding="utf-8"
    )

    print("=" * 78)
    print("PAPER NUMBER AUDIT: PASS")
    print("=" * 78)
    print(f"Artifacts: {output_dir}")
    print(
        "Combined split: "
        f"{int(validation.loc[validation['case'] == 'COMBINED', 'n_development'].iloc[0])}/"
        f"{int(validation.loc[validation['case'] == 'COMBINED', 'n_holdout'].iloc[0])}"
    )
    print(
        "Holdout platform separation: "
        f"{platform.loc[platform['population'] == 'COMBINED_UNTOUCHED_HOLDOUT', 'platform_separation_pct'].iloc[0]:.6f}%"
    )
    for row in routing.itertuples(index=False):
        print(
            f"{row.metric_id}: {row.value_pct:.6f}% "
            f"(N={row.n_population})"
        )
    print(
        "Hub-removal same-route rate: "
        f"{topology.iloc[0]['same_route_pct_after_removal']:.6f}%"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce reviewer-facing paper numbers"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Audit artifact directory (default: output/results/paper_audit)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_audit(output_dir.resolve())


if __name__ == "__main__":
    main()

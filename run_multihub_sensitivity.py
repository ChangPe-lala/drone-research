"""Run and aggregate the five one-factor-at-a-time graph configurations."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multihub_config import (
    GRAPH_SENSITIVITY_CONFIGS,
    ROUTING_POLICY_SCHEMA_VERSION,
)
from config.multiuav_config import VALIDATION_CASES
from config.paths import DEFAULT_OUTPUT_DIR


def _config_id(anchor: float, cluster: float) -> str:
    return f"anchor_{anchor:g}_cluster_{cluster:g}".replace(".", "p")


def _resolve_case(case: str | None, reference_output: Path) -> str:
    if case is not None:
        return case
    selected_path = (
        reference_output
        / "results"
        / "multiuav"
        / "validation"
        / "selected_configuration.json"
    )
    with selected_path.open("r", encoding="utf-8") as stream:
        selected = json.load(stream)
    selected_case = selected.get("selected_case")
    if selected_case not in VALIDATION_CASES:
        raise ValueError(f"Invalid selected case in {selected_path}: {selected_case}")
    return str(selected_case)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def _read_run(run_root: Path) -> dict[str, object]:
    data_dir = run_root / "data" / "multihub"
    result_dir = run_root / "results" / "multihub"
    nodes = pd.read_csv(data_dir / "graph_nodes.csv")
    edges = pd.read_csv(data_dir / "graph_edges.csv")
    scope = pd.read_csv(data_dir / "routing_experiment_scope.csv").iloc[0]
    hub = pd.read_csv(
        result_dir / "topology_audit" / "dominant_hub_removal_summary.csv"
    ).iloc[0]
    fair = pd.read_csv(
        result_dir / "final_rq5" / "fair_model_based_routing_comparison.csv"
    )
    strategies = pd.read_csv(result_dir / "route_baseline_comparison.csv")
    weight_summary = pd.read_csv(
        result_dir / "multi_objective_weight_sensitivity_summary.csv"
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "scope": scope,
        "hub": hub,
        "fair": fair,
        "strategies": strategies,
        "weight_summary": weight_summary,
    }


def _manifest_is_current(manifest_path: Path) -> bool:
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("routing_policy_schema_version")
        == ROUTING_POLICY_SCHEMA_VERSION
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-config Multi-Hub graph sensitivity analysis"
    )
    parser.add_argument("--case", choices=VALIDATION_CASES, default=None)
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR)
        / "models"
        / "multihub"
        / "segment_energy_model.pkl",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse configurations with a complete variant manifest.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Optional config ID such as anchor_200_cluster_15; repeat as needed.",
    )
    args = parser.parse_args()

    reference_output = args.reference_output.resolve()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    case = _resolve_case(args.case, reference_output)
    sensitivity_root = reference_output / "sensitivity_runs"
    aggregate_dir = reference_output / "results" / "multihub" / "sensitivity"
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = [
        (anchor, cluster)
        for anchor, cluster in GRAPH_SENSITIVITY_CONFIGS
        if not args.only or _config_id(anchor, cluster) in set(args.only)
    ]
    if not selected_configs:
        raise ValueError("No sensitivity configurations selected.")

    for anchor, cluster in selected_configs:
        config_id = _config_id(anchor, cluster)
        run_root = sensitivity_root / config_id
        manifest = run_root / "results" / "multihub" / "variant_manifest.json"
        if args.skip_existing and _manifest_is_current(manifest):
            print(f"Reusing completed sensitivity run: {config_id}")
            continue

        environment = os.environ.copy()
        environment.update(
            {
                "RS_OUTPUT_DIR": str(run_root),
                "RS_REFERENCE_OUTPUT_DIR": str(reference_output),
                "RS_SEGMENT_MODEL_PATH": str(model_path),
                "RS_HUB_INTERVAL_M": str(anchor),
                "RS_HUB_CLUSTER_M": str(cluster),
            }
        )
        print(
            f"Running {config_id}: anchor={anchor:g} m, "
            f"cluster={cluster:g} m"
        )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_multihub_variant.py"),
                "--case",
                case,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

    # Aggregation requires the baseline plus every configuration being reported.
    completed: dict[str, dict[str, object]] = {}
    for anchor, cluster in GRAPH_SENSITIVITY_CONFIGS:
        config_id = _config_id(anchor, cluster)
        run_root = sensitivity_root / config_id
        manifest = run_root / "results" / "multihub" / "variant_manifest.json"
        if _manifest_is_current(manifest):
            completed[config_id] = _read_run(run_root)

    baseline_id = _config_id(200.0, 15.0)
    if baseline_id not in completed:
        print("Baseline is not complete yet; per-config outputs were retained.")
        return

    baseline_fair = completed[baseline_id]["fair"]
    baseline_decisions = baseline_fair[
        ["mission_id", "recommendation_decision"]
    ].rename(columns={"recommendation_decision": "baseline_decision"})
    baseline_nodes = completed[baseline_id]["nodes"]

    summary_rows: list[dict[str, object]] = []
    consistency_rows: list[dict[str, object]] = []
    hub_rows: list[dict[str, object]] = []
    for anchor, cluster in GRAPH_SENSITIVITY_CONFIGS:
        config_id = _config_id(anchor, cluster)
        if config_id not in completed:
            continue
        result = completed[config_id]
        scope = result["scope"]
        hub = result["hub"]
        fair = result["fair"]
        strategies = result["strategies"]
        comparison = baseline_decisions.merge(
            fair[["mission_id", "recommendation_decision"]],
            on="mission_id",
            how="inner",
        )
        same = (
            comparison["baseline_decision"]
            == comparison["recommendation_decision"]
        )
        consistency = float(same.mean()) if len(same) else float("nan")
        consistency_rows.append(
            {
                "configuration": config_id,
                "n_baseline_missions": int(len(baseline_decisions)),
                "n_variant_missions": int(len(fair)),
                "n_comparable_scenarios": int(len(comparison)),
                "comparable_coverage_rate": (
                    float(len(comparison) / len(baseline_decisions))
                    if len(baseline_decisions)
                    else float("nan")
                ),
                "n_same_decision": int(same.sum()),
                "recommendation_consistency": consistency,
            }
        )

        nodes = result["nodes"]
        dominant_id = str(hub["dominant_hub"])
        dominant_row = nodes[nodes["node_id"].astype(str) == dominant_id].iloc[0]
        distances = baseline_nodes.apply(
            lambda row: _haversine_m(
                float(dominant_row["position_x"]),
                float(dominant_row["position_y"]),
                float(row["position_x"]),
                float(row["position_y"]),
            ),
            axis=1,
        )
        nearest_index = distances.idxmin()
        region_id = f"REGION_{baseline_nodes.loc[nearest_index, 'node_id']}"
        hub_rows.append(
            {
                "configuration": config_id,
                "dominant_hub": dominant_id,
                "dominant_hub_region": region_id,
                "dominant_hub_longitude": float(dominant_row["position_x"]),
                "dominant_hub_latitude": float(dominant_row["position_y"]),
                "distance_to_baseline_region_m": float(distances.loc[nearest_index]),
                "hub_selection_rate": float(
                    hub["dominant_hub_recommendation_frequency"]
                ),
            }
        )
        summary_rows.append(
            {
                "configuration": config_id,
                "anchor_interval_m": anchor,
                "cluster_threshold_m": cluster,
                "nodes": int(len(result["nodes"])),
                "edges": int(len(result["edges"])),
                "routable_od_pairs": int(scope["n_scored_topology_od_pairs"]),
                "routing_scenarios": int(scope["n_routing_scenarios"]),
                "candidate_routes": int(scope["n_candidate_routes"]),
                "fair_model_comparisons": int(len(fair)),
                "historical_strategy_comparisons": int(len(strategies)),
                "dominant_hub": dominant_id,
                "hub_selection_rate": float(
                    hub["dominant_hub_recommendation_frequency"]
                ),
                "recommendation_consistency": consistency,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    consistency_df = pd.DataFrame(consistency_rows)
    hub_df = pd.DataFrame(hub_rows)
    summary_df.to_csv(aggregate_dir / "topology_stability.csv", index=False)
    consistency_df.to_csv(
        aggregate_dir / "recommendation_stability.csv", index=False
    )
    hub_df.to_csv(aggregate_dir / "hub_stability.csv", index=False)
    def records_without_nan(frame: pd.DataFrame) -> list[dict[str, object]]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict(orient="records")

    payload = {
        "design": "one-factor-at-a-time",
        "baseline_configuration": baseline_id,
        "n_configurations": int(len(summary_df)),
        "topology_stability": records_without_nan(summary_df),
        "recommendation_stability": records_without_nan(consistency_df),
        "hub_stability": records_without_nan(hub_df),
    }
    with (aggregate_dir / "sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, default=str, allow_nan=False)
    print(f"Sensitivity summary saved to {aggregate_dir}")


if __name__ == "__main__":
    main()

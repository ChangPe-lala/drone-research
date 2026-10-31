"""Run one isolated graph-construction configuration with a frozen ML model.

This script is normally launched by ``run_multihub_sensitivity.py`` in a fresh
process so environment-scoped anchor, clustering, and output settings cannot
leak between configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multihub_config import (
    HUB_DISTANCE_THRESHOLD_M,
    ON_ROUTE_HUB_INTERVAL_M,
    ROUTING_POLICY_SCHEMA_VERSION,
)
from config.multiuav_config import VALIDATION_CASES
from config.paths import OUTPUT_DIR
from src.multihub.build_edge_request_features import build_edge_request_features
from src.multihub.build_route_graph import build_route_graph
from src.multihub.build_route_requests import build_route_requests
from src.multihub.build_segment_features import build_segment_features
from src.multihub.build_trajectory_index import build_trajectory_index
from src.multihub.cluster_hub_nodes import cluster_hub_nodes
from src.multihub.detect_hub_nodes import detect_hub_nodes
from src.multihub.enumerate_candidate_paths import enumerate_candidate_paths
from src.multihub.evaluate_routing import evaluate_routing
from src.multihub.final_ee_recommendation import build_final_ee_recommendations
from src.multihub.load_multihub_data import load_and_normalize_data
from src.multihub.map_trajectory_nodes import map_trajectory_nodes
from src.multihub.predict_edge_energy import predict_edge_energy
from src.multihub.routing_policies import run_routing_policies
from src.multihub.strategy_comparison import build_and_save_strategy_comparisons
from src.multihub.segment_trajectories import segment_trajectories
from src.multihub.topology_audit import run_topology_audit
from src.multihub.validate_segment_dataset import validate_segment_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="One Multi-Hub sensitivity run")
    parser.add_argument("--case", choices=VALIDATION_CASES, required=True)
    args = parser.parse_args()

    model_path = os.environ.get("RS_SEGMENT_MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            "RS_SEGMENT_MODEL_PATH must reference the frozen baseline segment model."
        )

    data = load_and_normalize_data(args.case)
    build_trajectory_index(data)
    candidates = detect_hub_nodes(data)
    nodes = cluster_hub_nodes(candidates)
    mapping = map_trajectory_nodes(data, nodes)
    segments = segment_trajectories(data, mapping)
    features = build_segment_features(segments)
    validate_segment_dataset(features)
    build_route_graph()
    build_route_requests()
    build_edge_request_features()
    predict_edge_energy()
    enumerate_candidate_paths()
    run_routing_policies()
    evaluate_routing()
    _, observed_baseline = build_final_ee_recommendations(args.case)
    build_and_save_strategy_comparisons(observed_baseline)
    run_topology_audit()

    result_dir = Path(OUTPUT_DIR) / "results" / "multihub"
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "validation_case": args.case,
        "anchor_interval_m": ON_ROUTE_HUB_INTERVAL_M,
        "cluster_threshold_m": HUB_DISTANCE_THRESHOLD_M,
        "output_dir": str(OUTPUT_DIR),
        "frozen_segment_model": str(Path(model_path).resolve()),
        "routing_policy_schema_version": ROUTING_POLICY_SCHEMA_VERSION,
    }
    with (result_dir / "variant_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()

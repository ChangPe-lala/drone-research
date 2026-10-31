
"""Shared Multi-Hub pipeline orchestration.

This module coordinates the existing research stages.
Energy calculations, models and routing policies remain unchanged.
"""

from src.multihub.load_multihub_data import load_and_normalize_data
from src.multihub.build_trajectory_index import build_trajectory_index
from src.multihub.detect_hub_nodes import detect_hub_nodes
from src.multihub.cluster_hub_nodes import cluster_hub_nodes
from src.multihub.map_trajectory_nodes import map_trajectory_nodes
from src.multihub.segment_trajectories import segment_trajectories
from src.multihub.build_segment_features import build_segment_features
from src.multihub.validate_segment_dataset import validate_segment_dataset

from src.multihub.build_route_graph import build_route_graph
from src.multihub.build_route_requests import build_route_requests
from src.multihub.build_edge_request_features import build_edge_request_features
from src.multihub.predict_edge_energy import predict_edge_energy
from src.multihub.enumerate_candidate_paths import enumerate_candidate_paths
from src.multihub.routing_policies import run_routing_policies
from src.multihub.evaluate_routing import evaluate_routing

from src.multihub.final_ee_recommendation import (
    build_final_ee_recommendations,
)
from src.multihub.strategy_comparison import (
    build_and_save_strategy_comparisons,
)
from src.multihub.topology_audit import run_topology_audit


def prepare_segment_data(case=None, *, check_nonempty=True):
    """Execute G00-G06 and validate the generated segments."""

    data = load_and_normalize_data(case)          # G00
    build_trajectory_index(data)                   # G01

    candidates = detect_hub_nodes(data)            # G02
    nodes = cluster_hub_nodes(candidates)          # G03
    mapping = map_trajectory_nodes(data, nodes)    # G04

    segments = segment_trajectories(data, mapping) # G05

    if check_nonempty and segments.empty:
        raise RuntimeError("G05 produced no valid segments.")

    features = build_segment_features(segments)    # G06
    validate_segment_dataset(features)

    return features


def run_routing_pipeline(
    case=None,
    *,
    include_graph=True,
    include_final_reports=True,
    include_topology_audit=True,
):
    """Execute routing stages and their optional reports."""

    if include_graph:
        print("G08 - Build route graph")
        build_route_graph()

    print("G09 - Build route requests")
    build_route_requests()

    print("G10 - Build edge/request features")
    build_edge_request_features()

    print("G11 - Predict edge energy")
    predict_edge_energy()

    print("G12 - Enumerate candidate paths")
    enumerate_candidate_paths()

    print("G13 - Apply routing policies")
    run_routing_policies()

    print("G14 - Evaluate routing")
    evaluate_routing()

    if include_final_reports:
        print("Final RQ5 recommendations")
        _, observed_baseline = build_final_ee_recommendations(case)

        print("Compare P0-P4 on identical missions")
        build_and_save_strategy_comparisons(observed_baseline)
    else:
        build_and_save_strategy_comparisons()

    if include_topology_audit:
        print("Topology audit")
        run_topology_audit()

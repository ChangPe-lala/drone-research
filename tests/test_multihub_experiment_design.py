import pandas as pd

from src.multihub.routing_graph import (
    build_supported_graph,
    enumerate_eligible_od_pairs,
    enumerate_feasible_paths,
    node_type_lookup,
)


def _graph_frames():
    nodes = pd.DataFrame(
        [
            {"node_id": "A", "node_type": "START"},
            {"node_id": "B", "node_type": "HUB"},
            {"node_id": "C", "node_type": "HUB"},
            {"node_id": "X", "node_type": "HUB"},
            {"node_id": "D", "node_type": "DESTINATION"},
        ]
    )
    rows = [
        ("A_B", "A", "B", 1.0, "Quadrotor"),
        ("B_D", "B", "D", 1.0, "Quadrotor"),
        ("A_C", "A", "C", 1.0, "Quadrotor"),
        ("C_D", "C", "D", 1.2, "Quadrotor"),
        ("A_X", "A", "X", 1.0, "Fixed-wing"),
        ("X_D", "X", "D", 1.0, "Fixed-wing"),
    ]
    edges = pd.DataFrame(
        [
            {
                "edge_id": edge_id,
                "start_node": start,
                "end_node": end,
                "segment_distance": distance,
                "supporting_uav_types": platform,
                "supporting_dataset_ids": "TEST",
            }
            for edge_id, start, end, distance, platform in rows
        ]
    )
    return nodes, edges


def test_od_expansion_uses_platform_hub_and_two_path_rules():
    nodes, edges = _graph_frames()
    graph = build_supported_graph(
        edges,
        uav_type="Quadrotor",
        dataset_id="TEST",
    )
    types = node_type_lookup(nodes)
    paths = enumerate_feasible_paths(
        graph,
        "A",
        "D",
        types=types,
        max_hops=4,
        max_distance_ratio=1.5,
    )
    assert [path for path, _ in paths] == [["A", "B", "D"], ["A", "C", "D"]]
    assert "X" not in graph

    eligible = enumerate_eligible_od_pairs(
        graph,
        types=types,
        max_hops=4,
        max_distance_ratio=1.5,
    )
    assert any(
        row["source_node"] == "A"
        and row["destination_node"] == "D"
        and row["n_candidate_paths"] == 2
        for row in eligible
    )


def test_direct_path_sets_the_detour_reference_but_is_not_a_multihub_candidate():
    nodes, edges = _graph_frames()
    direct = pd.DataFrame(
        [
            {
                "edge_id": "A_D",
                "start_node": "A",
                "end_node": "D",
                "segment_distance": 1.0,
                "supporting_uav_types": "Quadrotor",
                "supporting_dataset_ids": "TEST",
            }
        ]
    )
    graph = build_supported_graph(
        pd.concat([edges, direct], ignore_index=True),
        uav_type="Quadrotor",
        dataset_id="TEST",
    )
    paths = enumerate_feasible_paths(
        graph,
        "A",
        "D",
        types=node_type_lookup(nodes),
        max_hops=4,
        max_distance_ratio=1.5,
    )
    assert paths == []

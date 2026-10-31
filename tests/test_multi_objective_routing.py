"""Tests for the reviewer-facing P0/P1/P2/P3/P4 comparison."""

import pandas as pd
import pytest

from src.multihub.routing_policies import (
    score_multi_objective_paths,
    select_routing_policies,
)
from src.multihub.strategy_comparison import (
    build_observed_route_baseline_validation,
    build_route_baseline_comparison,
    build_route_baseline_summary,
)


def _requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_id": "REQ_1",
                "request_kind": "HISTORICAL_MISSION",
                "od_pair_id": "A__D",
                "source_flight": "MISSION_1",
                "source_node": "A",
                "destination_node": "D",
            }
        ]
    )


def _paths() -> pd.DataFrame:
    rows = [
        # Recorded Historical route (P0), scored by the same model.
        ("P_HIST", "A>H>D", 130.0, 11.0, 11.8, True),
        # Shortest and minimum-Energy alternative.
        ("P_LOW", "A>B>D", 100.0, 10.0, 10.0, False),
        # Maximum-EE alternative, with greater absolute Energy.
        ("P_EE", "A>C>D", 180.0, 12.0, 15.0, False),
        # Interior trade-off alternative.
        ("P_BAL", "A>X>D", 145.0, 10.8, 13.0, False),
    ]
    return pd.DataFrame(
        [
            {
                "request_id": "REQ_1",
                "request_kind": "HISTORICAL_MISSION",
                "od_pair_id": "A__D",
                "path_id": path_id,
                "node_sequence": route,
                "total_distance": distance,
                "predicted_total_energy": energy,
                "predicted_ee": ee,
                "path_energy_is_valid": True,
                "path_platform_match": True,
                "candidate_eligible": True,
                "is_original_path": is_original,
            }
            for path_id, route, distance, energy, ee, is_original in rows
        ]
    )


def test_policy_selection_contains_all_four_strategies():
    selection = select_routing_policies(_requests(), _paths()).iloc[0]

    assert selection["P1_SHORTEST_PATH_ID"] == "P_LOW"
    assert selection["P2_MIN_ENERGY_PATH_ID"] == "P_LOW"
    assert selection["P3_MAX_EE_PATH_ID"] == "P_EE"
    assert selection["P4_MULTI_OBJECTIVE_PATH_ID"] == "P_BAL"
    assert 0.0 <= selection["P4_MULTI_OBJECTIVE_SCORE"] <= 1.0


def test_multi_objective_weights_shift_the_selected_route():
    energy_first = select_routing_policies(
        _requests(), _paths(), ee_weight=0.25, energy_weight=0.75
    ).iloc[0]
    ee_first = select_routing_policies(
        _requests(), _paths(), ee_weight=0.75, energy_weight=0.25
    ).iloc[0]

    assert energy_first["P4_MULTI_OBJECTIVE_PATH_ID"] == "P_LOW"
    assert ee_first["P4_MULTI_OBJECTIVE_PATH_ID"] == "P_EE"


def test_multi_objective_weights_must_form_a_convex_combination():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        score_multi_objective_paths(
            _paths(), ee_weight=0.7, energy_weight=0.7
        )


def test_route_baseline_tables_place_p0_p4_on_the_same_mission():
    requests = _requests()
    paths = _paths()
    selections = select_routing_policies(requests, paths)
    observed = pd.DataFrame(
        [
            {
                "mission_id": "MISSION_1",
                "request_id": "REQ_1",
                "observed_historical_distance": 128.0,
                "observed_historical_energy": 10.5,
                "observed_historical_ee": 12.19,
            }
        ]
    )
    comparison = build_route_baseline_comparison(
        requests, paths, selections, observed
    )

    assert comparison["mission_id"].tolist() == ["MISSION_1"]
    expected = {
        "historical_route",
        "historical_distance",
        "historical_predicted_energy",
        "historical_predicted_ee",
        "historical_observed_energy",
        "historical_observed_ee",
        "shortest_route",
        "shortest_distance",
        "shortest_predicted_energy",
        "shortest_predicted_ee",
        "min_energy_route",
        "min_energy_distance",
        "min_energy_predicted_energy",
        "min_energy_predicted_ee",
        "max_ee_route",
        "max_ee_distance",
        "max_ee_predicted_energy",
        "max_ee",
        "multi_objective_route",
        "multi_objective_predicted_energy",
        "multi_objective_predicted_ee",
        "shortest_energy_saving_vs_historical_pct",
        "shortest_ee_gain_vs_historical_pct",
        "min_energy_saving_vs_historical_pct",
        "max_ee_energy_saving_vs_historical_pct",
        "max_ee_gain_vs_historical_pct",
    }
    assert expected.issubset(comparison.columns)
    summary = build_route_baseline_summary(comparison)
    assert summary["strategy"].tolist() == [
        "Historical",
        "Shortest-distance",
        "Minimum-energy",
        "Maximum-EE",
        "Multi-objective",
    ]
    assert summary.iloc[0]["mean_energy_saving_vs_historical_pct"] == 0.0
    support = build_observed_route_baseline_validation(comparison)
    assert len(support) == 1
    assert "max_ee_observed_baseline_ee_gain_pct" in support.columns

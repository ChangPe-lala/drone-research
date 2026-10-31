"""Canonical final Multi-Hub runner after CP3 and segment-model selection.

This executes graph/routing stages G08-G14 and then produces the final RQ5
historical-vs-hypothetical EE recommendation table.

Prerequisites
-------------
1. select_validated_configuration.py --case ...
2. run_multihub_g00_g07.py [--case ...]
3. fit_segment_energy_model.py --model ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import VALIDATION_CASES
from src.multihub.build_edge_request_features import build_edge_request_features
from src.multihub.build_route_graph import build_route_graph
from src.multihub.build_route_requests import build_route_requests
from src.multihub.enumerate_candidate_paths import enumerate_candidate_paths
from src.multihub.evaluate_routing import evaluate_routing
from src.multihub.final_ee_recommendation import build_final_ee_recommendations
from src.multihub.predict_edge_energy import predict_edge_energy
from src.multihub.routing_policies import run_routing_policies
from src.multihub.strategy_comparison import build_and_save_strategy_comparisons
from src.multihub.topology_audit import run_topology_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Final Multi-Hub EE analysis")
    parser.add_argument("--case", choices=VALIDATION_CASES, default=None,
                        help="Optional explicit case; otherwise use CP3 selection")
    args = parser.parse_args()

    print("G08 — graph")
    build_route_graph()
    print("G09 — historical requests + expanded topology OD scenarios")
    build_route_requests()
    print("G10 — edge/request features")
    build_edge_request_features()
    print("G11 — segment Energy prediction")
    predict_edge_energy()
    print("G12 — candidate paths")
    enumerate_candidate_paths()
    print("G13 — supporting routing policies")
    run_routing_policies()
    print("G14 — supporting energy-routing evaluation")
    evaluate_routing()

    print("FINAL RQ5 — fair model/model + observed-baseline support")
    _, observed_baseline = build_final_ee_recommendations(args.case)

    print("ROUTE BASELINES — P0/P1/P2/P3/P4 on identical missions")
    build_and_save_strategy_comparisons(observed_baseline)

    print("TOPOLOGY AUDIT — centrality, participation, dominant-hub removal")
    run_topology_audit()


if __name__ == "__main__":
    main()

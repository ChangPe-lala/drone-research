import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.multihub.build_route_requests import build_route_requests
from src.multihub.build_edge_request_features import build_edge_request_features
from src.multihub.predict_edge_energy import predict_edge_energy
from src.multihub.enumerate_candidate_paths import enumerate_candidate_paths
from src.multihub.routing_policies import run_routing_policies
from src.multihub.evaluate_routing import evaluate_routing
from src.multihub.strategy_comparison import build_and_save_strategy_comparisons

def main():
    print("="*60)
    print("STARTING ROUTING PIPELINE (G09 - G14)")
    print("="*60)
    
    # G09: Build route requests (Historical original path is strictly preserved)
    build_route_requests()
    
    # G10: Cross requests and edges to build features matching exact G07 schema (11 req + 1 edge)
    build_edge_request_features()
    
    # G11: Predict Edge Energy (Flag NaN/Inf/<=0 without clipping)
    predict_edge_energy()
    
    # G12: Enumerate Candidate Paths (up to MAX_PATH_HOPS) and calculate routing metrics
    enumerate_candidate_paths()
    
    # G13: Apply Routing Policies (P0, P1, P2, P3, P4)
    run_routing_policies()
    
    # G14: Evaluate Routing (Compare Predicted Original vs Predicted Recommended)
    evaluate_routing()

    # Reviewer table: all four strategies evaluated on identical missions.
    build_and_save_strategy_comparisons()
    
    print("="*60)
    print("PIPELINE (G09 - G14) COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

"""
test_routing_policies.py
========================
Tests for multi-hub candidate builder and routing policies.
"""
import pytest
import pandas as pd
from src.routing.candidate_builder import build_hub_candidates, validate_candidate_features
from experiments.E12_multihub_policy import run_policy_p4

def test_candidate_builder_preflight_guard():
    schema = {
        "planned_distance": {"deployability": "PRE_FLIGHT"},
        "actual_speed": {"deployability": "POST_FLIGHT"}
    }
    
    # Should fail fast
    with pytest.raises(ValueError, match="FAIL FAST"):
        validate_candidate_features(["planned_distance", "actual_speed"], schema)
        
    # Should pass
    validate_candidate_features(["planned_distance"], schema)

def test_candidate_builder_uniqueness():
    schema = {
        "planned_distance": {"deployability": "PRE_FLIGHT"},
        "route_bearing": {"deployability": "PRE_FLIGHT"}
    }
    base_request = {}
    hubs = [
        {"id": "HubA", "distance_to_dest": 1000, "bearing_to_dest": 45},
        {"id": "HubB", "distance_to_dest": 1000, "bearing_to_dest": 45} # Identical
    ]
    
    with pytest.raises(ValueError, match="identical physical vectors"):
        build_hub_candidates(base_request, hubs, schema)

def test_p4_risk_policy():
    # Candidates with score components
    candidates = pd.DataFrame({
        "hub": ["A", "B", "C"],
        "ee_predicted_norm": [0.8, 0.9, 0.95],
        "uncertainty_norm": [0.2, 0.1, 0.8],
        "ood_normalized_score": [0.1, 0.2, 0.9] # Hub C is very OOD
    })
    
    # Hub C should be rejected by hard filter (e.g. threshold 0.8)
    # Between A and B:
    # Score A = 0.8 - (0.5 * 0.2) - (0.5 * 0.1) = 0.8 - 0.1 - 0.05 = 0.65
    # Score B = 0.9 - (0.5 * 0.1) - (0.5 * 0.2) = 0.9 - 0.05 - 0.1 = 0.75
    
    best = run_policy_p4(candidates, lambda_u=0.5, lambda_o=0.5, hard_ood_threshold=0.8)
    
    assert best["hub"] == "B"

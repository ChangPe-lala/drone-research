"""
test_battery_scheduler.py
=========================
Tests for explicit battery dispatch rules.
"""
import pytest
from src.routing.battery_scheduler import evaluate_dispatch_safety, get_required_energy_for_scheduler
from experiments.E13_battery_scheduling import run_s2_ood_ucb_scheduler

def test_evaluate_dispatch_safety():
    # Capacity = 100 Wh. SOC = 100% -> Available = 100 Wh
    # Reserve = 20% -> 20 Wh
    # Required = 70 Wh
    # Available (100) - Required (70) = 30 >= 20. Should be SAFE.
    assert evaluate_dispatch_safety(soc=1.0, capacity_wh=100.0, required_energy_wh=70.0, min_reserve_fraction=0.2) == True
    
    # Required = 85 Wh
    # Available (100) - Required (85) = 15 < 20. Should be UNSAFE.
    assert evaluate_dispatch_safety(soc=1.0, capacity_wh=100.0, required_energy_wh=85.0, min_reserve_fraction=0.2) == False
    
    # SOC = 50% -> Available = 50 Wh
    # Required = 30 Wh
    # Available (50) - Required (30) = 20 >= 20. Should be SAFE.
    assert evaluate_dispatch_safety(soc=0.5, capacity_wh=100.0, required_energy_wh=30.0, min_reserve_fraction=0.2) == True

def test_s2_ood_ucb_scheduler():
    capacity = 100.0
    reserve = 0.2
    
    # OOD very high
    request_ood = {
        "soc": 1.0,
        "energy_upper_bound": 50.0,
        "ood_normalized_score": 0.95
    }
    # Should reject explicitly
    assert run_s2_ood_ucb_scheduler(request_ood, capacity, reserve, hard_ood_threshold=0.8) == "REJECTED_OOD"
    
    # Normal request, safe battery
    request_normal_safe = {
        "soc": 1.0,
        "energy_upper_bound": 50.0,
        "ood_normalized_score": 0.2
    }
    assert run_s2_ood_ucb_scheduler(request_normal_safe, capacity, reserve, hard_ood_threshold=0.8) == "DISPATCHED"
    
    # Normal request, unsafe battery (Upper bound too high)
    request_normal_unsafe = {
        "soc": 1.0,
        "energy_upper_bound": 90.0, # 100 - 90 = 10 < 20
        "ood_normalized_score": 0.2
    }
    assert run_s2_ood_ucb_scheduler(request_normal_unsafe, capacity, reserve, hard_ood_threshold=0.8) == "DEFERRED_BATTERY"

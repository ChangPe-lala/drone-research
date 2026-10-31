"""
test_target_strategies.py
=========================
Validates the target derivation strategies for E07.
Ensures correct handling of invalid physical predictions and distance types.
"""
import pytest
import pandas as pd
import numpy as np
from src.modeling.target_strategies import derive_efficiency, DistanceContext

def test_derive_efficiency_valid():
    energy = pd.Series([10.0, 20.0])
    distance = pd.Series([1000.0, 500.0])
    
    result = derive_efficiency(energy, distance, DistanceContext.ACTUAL_DISTANCE)
    
    assert result['physical_invalid_prediction'].sum() == 0
    assert np.isclose(result['derived_ee'].iloc[0], 100.0)
    assert np.isclose(result['derived_ee'].iloc[1], 25.0)

def test_derive_efficiency_invalid_energy():
    energy = pd.Series([10.0, -5.0, 0.0, np.nan])
    distance = pd.Series([1000.0, 1000.0, 1000.0, 1000.0])
    
    result = derive_efficiency(energy, distance, DistanceContext.PLANNED_DISTANCE)
    
    invalid_flags = result['physical_invalid_prediction']
    
    assert invalid_flags.iloc[0] == False
    assert invalid_flags.iloc[1] == True  # Negative energy
    assert invalid_flags.iloc[2] == True  # Zero energy
    assert invalid_flags.iloc[3] == True  # NaN energy

def test_derive_efficiency_invalid_distance():
    energy = pd.Series([10.0, 10.0])
    distance = pd.Series([np.nan, np.inf])
    
    result = derive_efficiency(energy, distance, DistanceContext.ACTUAL_DISTANCE)
    
    invalid_flags = result['physical_invalid_prediction']
    
    assert invalid_flags.iloc[0] == True # EE will be NaN
    assert invalid_flags.iloc[1] == True # EE will be Inf
    
def test_derive_efficiency_enum_check():
    energy = pd.Series([10.0])
    distance = pd.Series([1000.0])
    
    with pytest.raises(ValueError):
        derive_efficiency(energy, distance, "ACTUAL") # string instead of enum

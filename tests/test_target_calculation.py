
import numpy as np

from src.data_processing.target_calculation import (
    calculate_battery_energy,
    compute_planned_wind_features,
    compute_route_direction,
)


def test_energy_excludes_long_gap():
    # 10 W constant power. Only the first 1 s is valid.
    v = np.array([10.0, 10.0, 10.0])
    i = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 1.0, 100.0])

    corrected = calculate_battery_energy(
        v, i, t, max_gap_sec=30.0
    )
    legacy = calculate_battery_energy(
        v, i, t, max_gap_sec=None
    )

    assert np.isclose(corrected, 10.0 / 3600.0)
    assert np.isclose(legacy, 1000.0 / 3600.0)


def test_energy_excludes_nonpositive_dt():
    v = np.array([10.0, 10.0, 10.0])
    i = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 0.0, 1.0])

    e = calculate_battery_energy(
        v, i, t, max_gap_sec=30.0
    )

    assert np.isclose(e, 10.0 / 3600.0)


def test_planned_wind_tailwind_and_headwind():
    # Eastbound route, wind FROM west -> tailwind.
    # 180 degrees = tailwind.
    rel, head, cross = compute_planned_wind_features(
        -79.9, 40.5,
        -79.8, 40.5,
        270.0, 10.0,
        coordinate_mode="GEOGRAPHIC",
    )

    assert np.isclose(rel, 180.0, atol=0.2)
    assert np.isclose(head, -10.0, atol=0.1)
    assert np.isclose(cross, 0.0, atol=0.1)

    # Westbound route, wind FROM west -> headwind.
    # 0 degrees = headwind.
    rel, head, cross = compute_planned_wind_features(
        -79.8, 40.5,
        -79.9, 40.5,
        270.0, 10.0,
        coordinate_mode="GEOGRAPHIC",
    )

    assert np.isclose(rel, 0.0, atol=0.2)
    assert np.isclose(head, 10.0, atol=0.1)
    assert np.isclose(cross, 0.0, atol=0.1)


def test_local_route_direction_compass_convention():
    assert np.isclose(
        compute_route_direction(
            0, 0, 0, 1, "LOCAL_XYZ"
        ),
        0.0,
    )

    assert np.isclose(
        compute_route_direction(
            0, 0, 1, 0, "LOCAL_XYZ"
        ),
        90.0,
    )

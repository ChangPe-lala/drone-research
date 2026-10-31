
import numpy as np
import pandas as pd

from src.data_processing.physics_feature_engineering import (
    compute_headwind,
    compute_crosswind,
    compute_moist_air_density,
    attach_physics_features,
)

from src.data_processing.target_calculation import (
    compute_planned_wind_features,
)


def test_wind_components_headwind_convention():
    w = pd.Series([10.0])

    # 0 degrees = headwind
    assert np.isclose(
        compute_headwind(
            w, pd.Series([0.0])
        ).iloc[0],
        10.0,
    )

    assert np.isclose(
        compute_crosswind(
            w, pd.Series([0.0])
        ).iloc[0],
        0.0,
        atol=1e-7,
    )

    # 90 degrees = crosswind
    assert np.isclose(
        compute_headwind(
            w, pd.Series([90.0])
        ).iloc[0],
        0.0,
        atol=1e-7,
    )

    assert np.isclose(
        compute_crosswind(
            w, pd.Series([90.0])
        ).iloc[0],
        10.0,
    )

    # 180 degrees = tailwind
    assert np.isclose(
        compute_headwind(
            w, pd.Series([180.0])
        ).iloc[0],
        -10.0,
    )

    assert np.isclose(
        compute_crosswind(
            w, pd.Series([180.0])
        ).iloc[0],
        0.0,
        atol=1e-7,
    )


def test_air_density_extremes():
    rho_0 = compute_moist_air_density(
        pd.Series([20.0]),
        pd.Series([0.0]),
        pd.Series([1013.25]),
    )

    rho_100 = compute_moist_air_density(
        pd.Series([20.0]),
        pd.Series([100.0]),
        pd.Series([1013.25]),
    )

    assert np.isfinite(rho_0.iloc[0])
    assert np.isfinite(rho_100.iloc[0])
    assert rho_100.iloc[0] < rho_0.iloc[0]


def test_attach_physics_preflight_columns():
    df = pd.DataFrame({
        "speed": [10.0, 20.0],
        "wind_speed_ms": [0.0, 0.0],
        "planned_relative_wind_angle": [0.0, 0.0],
        "wind_gust_ms": [0.0, 0.0],
        "temperature_c": [20.0, 20.0],
        "humidity_pct": [50.0, 50.0],
        "pressure_hpa": [1013.0, 1013.0],
    })

    out = attach_physics_features(df)

    numeric = out.select_dtypes(
        include=[np.number]
    ).values

    assert np.isfinite(numeric).all()

    assert (
        out["effective_airspeed"].iloc[1]
        > out["effective_airspeed"].iloc[0]
    )

    assert (
        out["aerodynamic_power_proxy"].iloc[1]
        > out["aerodynamic_power_proxy"].iloc[0]
    )


def test_wind_convention_consistency():
    # Eastbound route, wind FROM west -> tailwind.
    rel, expected_head, expected_cross = (
        compute_planned_wind_features(
            -79.9, 40.5,
            -79.8, 40.5,
            270.0, 10.0,
            coordinate_mode="GEOGRAPHIC",
        )
    )

    w = pd.Series([10.0])
    theta = pd.Series([rel])

    actual_head = compute_headwind(
        w, theta
    ).iloc[0]

    actual_cross = compute_crosswind(
        w, theta
    ).iloc[0]

    assert np.isclose(
        actual_head, expected_head
    )

    assert np.isclose(
        actual_cross, expected_cross
    )

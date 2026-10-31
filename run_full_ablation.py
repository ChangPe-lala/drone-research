"""
run_full_ablation.py
====================

Post-hoc diagnostic ablation for the FINAL CP3 validation configuration.

IMPORTANT
---------
This script does NOT redo feature selection and does NOT change CP1/CP2/CP3.
It is an explanatory/diagnostic experiment for RQ1/RQ2.

For the current final research configuration:
    CP3 case       : DJI_ONLY
    target         : energy_efficiency [m/Wh]
    selected model : read from selected_configuration.json (currently CatBoost)
    model params   : frozen CP2 params from selected_configuration.json

Ablation baseline:
    FULL = Operational + Environmental = 11 features

Group ablation:
    FULL
    Without Operational
    Without Environmental

Individual ablation:
    FULL minus each of the 11 features, one at a time.

Evaluation:
    Development partition ONLY
    Same deterministic development/holdout split as CP1/CP2
    Same GroupKFold implementation used by the validation pipeline
    Untouched holdout is NOT used here.

Delta definitions:
    Delta_R2 = R2_FULL - R2_ABLATED
        > 0 : removal worsened R2
        < 0 : removal improved R2

    Delta_WMAPE = WMAPE_ABLATED - WMAPE_FULL
        > 0 : removal worsened WMAPE
        < 0 : removal improved WMAPE

Outputs:
    output/results/ablation/dji_only/
        full_group_ablation.csv
        full_individual_ablation.csv
        full_ablation_fold_metrics.csv
        full_ablation_manifest.json

    output/figures/ablation/dji_only/
        group_ablation_delta_r2.png
        group_ablation_delta_wmape.png
        individual_ablation_delta_r2.png
        individual_ablation_delta_wmape.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import (
    CASE_DJI_ONLY,
    ENVIRONMENTAL_FEATURES,
    OPERATIONAL_FEATURES,
    PRIMARY_TARGET,
)
from config.paths import (
    OUT_FIG_ABLATION,
    OUT_RES_ABLATION,
    OUT_RES_MULTIUAV,
)
from src.modeling.metrics import compute_metrics
from src.modeling.validation_common import (
    development_holdout_split,
    evaluate_cv_configuration,
    load_case_dataset,
    save_json,
    summarize_fold_metrics,
)


SELECTION_PATH = (
    Path(OUT_RES_MULTIUAV)
    / "validation"
    / "selected_configuration.json"
)

RESULT_DIR = Path(OUT_RES_ABLATION) / "dji_only"
FIGURE_DIR = Path(OUT_FIG_ABLATION) / "dji_only"

FULL_FEATURES = list(OPERATIONAL_FEATURES) + list(ENVIRONMENTAL_FEATURES)


def _load_cp3_selection() -> dict:
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing CP3 selection: {SELECTION_PATH}\n"
            "Run: python select_validated_configuration.py --case DJI_ONLY"
        )

    with SELECTION_PATH.open("r", encoding="utf-8") as f:
        selection = json.load(f)

    selected_case = str(selection.get("selected_case", ""))
    if selected_case != CASE_DJI_ONLY:
        raise ValueError(
            "This post-hoc RQ1/RQ2 ablation is currently defined for the "
            f"selected DJI_ONLY configuration, but CP3 contains: {selected_case}"
        )

    model_name = selection.get("model")
    params = selection.get("best_params")

    if not model_name:
        raise ValueError(
            f"CP3 selection does not contain 'model': {SELECTION_PATH}"
        )
    if params is None:
        raise ValueError(
            f"CP3 selection does not contain 'best_params': {SELECTION_PATH}"
        )

    return selection


def _validate_dataset_columns(df: pd.DataFrame) -> None:
    required = FULL_FEATURES + [PRIMARY_TARGET, "flight_uid", "dataset_id", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "DJI dataset is missing columns required for full OPS+ENV ablation:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def _run_scenario(
    dev: pd.DataFrame,
    case: str,
    scenario: str,
    removed: list[str],
    kept: list[str],
    model_name: str,
    params: dict,
) -> tuple[dict, pd.DataFrame]:
    if not kept:
        raise ValueError(f"Scenario '{scenario}' has no remaining features.")

    fold_df, oof_df = evaluate_cv_configuration(
        dev,
        case,
        kept,
        model_name,
        params,
    )

    summary = summarize_fold_metrics(fold_df)
    oof_metrics = compute_metrics(
        oof_df["y_true"].to_numpy(float),
        oof_df["y_pred"].to_numpy(float),
    )

    row = {
        "Scenario": scenario,
        "Removed": ",".join(removed),
        "Kept_Features": ",".join(kept),
        "N_Features": len(kept),
        **summary,
        "OOF_R2": oof_metrics.get("R2", np.nan),
        "OOF_MAE": oof_metrics.get("MAE", np.nan),
        "OOF_RMSE": oof_metrics.get("RMSE", np.nan),
        "OOF_WMAPE": oof_metrics.get("WMAPE", np.nan),
        "OOF_Bias": oof_metrics.get("Bias", np.nan),
    }

    fold_out = fold_df.copy()
    fold_out.insert(0, "Scenario", scenario)
    fold_out.insert(1, "Removed", ",".join(removed))
    fold_out.insert(2, "N_Features", len(kept))

    return row, fold_out


def _add_deltas(rows: list[dict], full_row: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    df["Delta_R2"] = (
        float(full_row["CV_R2_mean"])
        - pd.to_numeric(df["CV_R2_mean"], errors="coerce")
    )
    df["Delta_WMAPE"] = (
        pd.to_numeric(df["CV_WMAPE_mean"], errors="coerce")
        - float(full_row["CV_WMAPE_mean"])
    )

    # Percentage-point version is easier to read in the paper.
    df["CV_WMAPE_pct"] = (
        pd.to_numeric(df["CV_WMAPE_mean"], errors="coerce") * 100.0
    )
    df["Delta_WMAPE_pp"] = (
        pd.to_numeric(df["Delta_WMAPE"], errors="coerce") * 100.0
    )

    return df


def _plot_horizontal(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    plot_df = df.copy()
    values = pd.to_numeric(plot_df[value_col], errors="coerce")
    valid = np.isfinite(values)
    plot_df = plot_df.loc[valid].copy()

    if plot_df.empty:
        return

    # Put largest values at the top.
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig_height = max(4.0, 0.48 * len(plot_df) + 1.8)
    plt.figure(figsize=(9.5, fig_height))
    plt.barh(
        plot_df[label_col].astype(str),
        plot_df[value_col].astype(float),
    )
    plt.axvline(0.0, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    selection = _load_cp3_selection()

    case = selection["selected_case"]
    model_name = selection["model"]
    params = dict(selection["best_params"])

    df, source_path = load_case_dataset(case)
    _validate_dataset_columns(df)

    # IMPORTANT: reproduce the exact deterministic split used earlier.
    dev, holdout, split_manifest = development_holdout_split(df, case)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("POST-HOC FULL ABLATION — RQ1 / RQ2")
    print("=" * 78)
    print(f"Case: {case}")
    print(f"Target: {PRIMARY_TARGET} [m/Wh]")
    print(f"Model: {model_name}")
    print(f"Frozen params: {params}")
    print(f"Development rows: {len(dev)}")
    print(f"Untouched holdout rows NOT USED: {len(holdout)}")
    print(f"Full features ({len(FULL_FEATURES)}): {FULL_FEATURES}")
    print("CV: same group-aware development CV as CP1/CP2")
    print("=" * 78)

    all_fold_frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # FULL baseline
    # ------------------------------------------------------------------
    full_row, full_folds = _run_scenario(
        dev=dev,
        case=case,
        scenario="FULL_OPS_ENV",
        removed=[],
        kept=FULL_FEATURES,
        model_name=model_name,
        params=params,
    )
    all_fold_frames.append(full_folds)

    # ------------------------------------------------------------------
    # GROUP ABLATION
    # ------------------------------------------------------------------
    group_rows = [full_row]

    group_specs = [
        (
            "WITHOUT_OPERATIONAL",
            list(OPERATIONAL_FEATURES),
            [f for f in FULL_FEATURES if f not in OPERATIONAL_FEATURES],
        ),
        (
            "WITHOUT_ENVIRONMENTAL",
            list(ENVIRONMENTAL_FEATURES),
            [f for f in FULL_FEATURES if f not in ENVIRONMENTAL_FEATURES],
        ),
    ]

    for scenario, removed, kept in group_specs:
        row, folds = _run_scenario(
            dev=dev,
            case=case,
            scenario=scenario,
            removed=removed,
            kept=kept,
            model_name=model_name,
            params=params,
        )
        group_rows.append(row)
        all_fold_frames.append(folds)

    group_df = _add_deltas(group_rows, full_row)

    # Helpful interpretation. This is predictive, not causal.
    group_df["Predictive_Interpretation"] = np.select(
        [
            group_df["Delta_R2"] > 0,
            group_df["Delta_R2"] < 0,
        ],
        [
            "Removal worsened CV R2",
            "Removal improved CV R2",
        ],
        default="Baseline/no material R2 change",
    )

    group_df.to_csv(
        RESULT_DIR / "full_group_ablation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # INDIVIDUAL ABLATION
    # ------------------------------------------------------------------
    individual_rows = []

    for feature in FULL_FEATURES:
        kept = [f for f in FULL_FEATURES if f != feature]
        scenario = f"WITHOUT_{feature}"

        row, folds = _run_scenario(
            dev=dev,
            case=case,
            scenario=scenario,
            removed=[feature],
            kept=kept,
            model_name=model_name,
            params=params,
        )
        row["Removed_Feature"] = feature
        row["Feature_Group"] = (
            "Operational"
            if feature in OPERATIONAL_FEATURES
            else "Environmental"
        )
        individual_rows.append(row)
        all_fold_frames.append(folds)

    individual_df = _add_deltas(individual_rows, full_row)
    individual_df["Predictive_Interpretation"] = np.select(
        [
            individual_df["Delta_R2"] > 0,
            individual_df["Delta_R2"] < 0,
        ],
        [
            "Removal worsened CV R2",
            "Removal improved CV R2",
        ],
        default="No material R2 change",
    )

    # Larger positive Delta_R2 = stronger positive predictive contribution.
    individual_df = individual_df.sort_values(
        ["Delta_R2", "Delta_WMAPE"],
        ascending=[False, False],
    )
    individual_df.to_csv(
        RESULT_DIR / "full_individual_ablation.csv",
        index=False,
    )

    # Save every fold for audit/reproducibility.
    pd.concat(all_fold_frames, ignore_index=True).to_csv(
        RESULT_DIR / "full_ablation_fold_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # FIGURES — no physical/causal claim, only predictive degradation.
    # ------------------------------------------------------------------
    group_plot = group_df[group_df["Scenario"] != "FULL_OPS_ENV"].copy()

    _plot_horizontal(
        group_plot,
        "Scenario",
        "Delta_R2",
        "Group Ablation: Change in CV R²",
        "ΔR² = R²(full) − R²(ablated)",
        FIGURE_DIR / "group_ablation_delta_r2.png",
    )

    _plot_horizontal(
        group_plot,
        "Scenario",
        "Delta_WMAPE_pp",
        "Group Ablation: Change in CV WMAPE",
        "ΔWMAPE (percentage points)",
        FIGURE_DIR / "group_ablation_delta_wmape.png",
    )

    _plot_horizontal(
        individual_df,
        "Removed_Feature",
        "Delta_R2",
        "Individual Feature Ablation: Change in CV R²",
        "ΔR² = R²(full) − R²(without feature)",
        FIGURE_DIR / "individual_ablation_delta_r2.png",
    )

    _plot_horizontal(
        individual_df,
        "Removed_Feature",
        "Delta_WMAPE_pp",
        "Individual Feature Ablation: Change in CV WMAPE",
        "ΔWMAPE (percentage points)",
        FIGURE_DIR / "individual_ablation_delta_wmape.png",
    )

    manifest = {
        "experiment": "post_hoc_full_ablation",
        "purpose": (
            "RQ1/RQ2 diagnostic analysis after CP3; "
            "not used to revise CP1/CP2/CP3 decisions"
        ),
        "case": case,
        "target": PRIMARY_TARGET,
        "target_unit": "m/Wh",
        "model": model_name,
        "frozen_params": params,
        "source_file": str(source_path),
        "full_features": FULL_FEATURES,
        "operational_features": list(OPERATIONAL_FEATURES),
        "environmental_features": list(ENVIRONMENTAL_FEATURES),
        "n_total_rows": int(len(df)),
        "n_development_rows": int(len(dev)),
        "n_holdout_rows_not_used": int(len(holdout)),
        "split_manifest": split_manifest,
        "evaluation": (
            "development-only group-aware CV via "
            "src.modeling.validation_common.evaluate_cv_configuration"
        ),
        "delta_r2_definition": "R2_full - R2_ablated",
        "delta_wmape_definition": "WMAPE_ablated - WMAPE_full",
        "important_interpretation_note": (
            "Ablation measures predictive contribution under this fixed model/CV. "
            "It does not establish physical causality."
        ),
    }
    save_json(
        manifest,
        RESULT_DIR / "full_ablation_manifest.json",
    )

    print("\nGROUP ABLATION")
    show_group_cols = [
        "Scenario",
        "N_Features",
        "CV_R2_mean",
        "CV_R2_std",
        "CV_WMAPE_pct",
        "Delta_R2",
        "Delta_WMAPE_pp",
    ]
    print(group_df[show_group_cols].to_string(index=False))

    print("\nINDIVIDUAL ABLATION — ranked by Delta_R2")
    show_ind_cols = [
        "Removed_Feature",
        "Feature_Group",
        "CV_R2_mean",
        "CV_WMAPE_pct",
        "Delta_R2",
        "Delta_WMAPE_pp",
    ]
    print(individual_df[show_ind_cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("FULL ABLATION COMPLETE")
    print(f"Results: {RESULT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
"""CP2 — Model benchmark/tuning after an explicit CP1 feature-set decision.

Examples
--------
python run_model_benchmark.py --case DJI_ONLY --feature-set FS_OPS_ENV
python run_model_benchmark.py --case COMBINED --feature-set FS_OPS_ENV_PLATFORM

CP2 tunes hyperparameters *within* each candidate model on the development
partition using the same group-aware CV protocol. Selecting the best
hyperparameters for each model is an algorithmic tuning step; selecting which
MODEL proceeds is a research decision and is never done automatically.

The untouched holdout created before CP1 remains unused here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import (
    MODEL_BENCHMARK_CANDIDATES,
    PRIMARY_TARGET,
    VALIDATION_CASES,
    get_validation_feature_set,
    validate_primary_ee_feature_list,
)
from src.modeling.validation_common import (
    MODEL_PARAM_GRIDS,
    case_result_dir,
    development_holdout_split,
    evaluate_cv_configuration,
    load_case_dataset,
    parse_model_list,
    save_json,
    summarize_fold_metrics,
)


def _selection_key(row: dict) -> tuple:
    """Within one model: high R2, low WMAPE, low R2 fold std."""
    r2 = row.get("CV_R2_mean", np.nan)
    wmape = row.get("CV_WMAPE_mean", np.nan)
    r2_std = row.get("CV_R2_std", np.nan)
    return (
        -np.inf if not np.isfinite(r2) else r2,
        -np.inf if not np.isfinite(wmape) else -wmape,
        -np.inf if not np.isfinite(r2_std) else -r2_std,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CP2 model benchmark")
    parser.add_argument("--case", required=True, choices=VALIDATION_CASES)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--models", default="all", help="Comma-separated models or 'all'.")
    args = parser.parse_args()

    case = args.case
    features = get_validation_feature_set(args.feature_set, case)
    validate_primary_ee_feature_list(features)
    model_names = parse_model_list(args.models, MODEL_BENCHMARK_CANDIDATES)

    df, source_path = load_case_dataset(case)
    dev, holdout, split_manifest = development_holdout_split(df, case)
    missing = [f for f in features if f not in dev.columns]
    if missing:
        raise ValueError(f"Missing selected features: {missing}")

    out_dir = case_result_dir(case) / "cp2_model_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_rows = []
    best_rows = []
    best_fold_frames = []
    best_oof_frames = []

    print("=" * 78)
    print(f"CP2 MODEL BENCHMARK — {case}")
    print(f"Feature set: {args.feature_set} ({len(features)} features)")
    print(f"Primary target: {PRIMARY_TARGET} [m/Wh]")
    print(f"Development: {len(dev)} | Untouched holdout: {len(holdout)}")
    print("=" * 78)

    for model_name in model_names:
        params_list = MODEL_PARAM_GRIDS[model_name]
        print(f"\n[{model_name}] {len(params_list)} hyperparameter configuration(s)")
        model_configs = []

        for config_id, params in enumerate(params_list, start=1):
            fold_df, _ = evaluate_cv_configuration(
                dev, case, features, model_name, params
            )
            summary = summarize_fold_metrics(fold_df)
            row = {
                "Model": model_name,
                "Config_ID": config_id,
                "Params_JSON": json.dumps(params, sort_keys=True),
                **summary,
            }
            config_rows.append(row)
            model_configs.append(row)

        # Automatic choice is only among hyperparameters of the SAME model.
        best = max(model_configs, key=_selection_key)
        best_params = json.loads(best["Params_JSON"])
        fold_df, oof_df = evaluate_cv_configuration(
            dev, case, features, model_name, best_params
        )
        fold_df.insert(0, "Model", model_name)
        oof_df.insert(0, "Model", model_name)
        best_fold_frames.append(fold_df)
        best_oof_frames.append(oof_df)

        best_rows.append(
            {
                "Case": case,
                "Feature_Set": args.feature_set,
                "Model": model_name,
                "Best_Params_JSON": json.dumps(best_params, sort_keys=True),
                "CV_R2_mean": best["CV_R2_mean"],
                "CV_R2_std": best["CV_R2_std"],
                "CV_MAE_mean": best["CV_MAE_mean"],
                "CV_RMSE_mean": best["CV_RMSE_mean"],
                "CV_WMAPE_mean": best["CV_WMAPE_mean"],
                "CV_WMAPE_std": best["CV_WMAPE_std"],
                "N_Development": len(dev),
                "N_Holdout_Unused": len(holdout),
            }
        )
        print(
            f"  best params={best_params} | "
            f"R2={best['CV_R2_mean']:.4f} | "
            f"WMAPE={best['CV_WMAPE_mean']*100:.2f}%"
        )

    config_df = pd.DataFrame(config_rows)
    evidence_df = pd.DataFrame(best_rows)
    config_df.to_csv(out_dir / "hyperparameter_evidence.csv", index=False)
    evidence_df.to_csv(out_dir / "model_evidence.csv", index=False)
    pd.concat(best_fold_frames, ignore_index=True).to_csv(
        out_dir / "best_config_fold_metrics.csv", index=False
    )
    pd.concat(best_oof_frames, ignore_index=True).to_csv(
        out_dir / "best_config_oof_predictions.csv", index=False
    )

    save_json(
        {
            "checkpoint": "CP2_MODEL_SELECTION",
            "case": case,
            "feature_set": args.feature_set,
            "features": features,
            "primary_target": PRIMARY_TARGET,
            "source_file": str(source_path),
            "models_evaluated": model_names,
            "split": split_manifest,
            "holdout_used_for_cp2": False,
            "automatic_model_selection": False,
            "note": (
                "Best hyperparameters are selected within each model by development CV; "
                "the model family itself is not selected automatically."
            ),
        },
        out_dir / "run_manifest.json",
    )

    (out_dir / "CHECKPOINT_CP2.txt").write_text(
        f"""CP2 MODEL SELECTION — {case}\n\nModel evidence has been generated using {args.feature_set}.\nThe untouched holdout has NOT been used.\nNo model family was selected automatically.\n\nReview:\n  {out_dir / 'model_evidence.csv'}\n\nThen continue explicitly, for example:\n  python finalize_case.py --case {case} --feature-set {args.feature_set} --model ExtraTrees\n""",
        encoding="utf-8",
    )

    print("\n" + evidence_df.to_string(index=False))
    print("\nCP2 COMPLETE — STOPPING FOR MODEL DECISION")


if __name__ == "__main__":
    main()

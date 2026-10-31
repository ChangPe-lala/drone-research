"""CP1 — Feature-set screening for primary flight-level Energy Efficiency.

Usage
-----
python run_feature_screen.py --case DJI_ONLY
python run_feature_screen.py --case VTOL_ONLY
python run_feature_screen.py --case COMBINED

The runner benchmarks only scientifically eligible feature sets on the
DEVELOPMENT partition. The untouched group holdout is created before CP1 and
is never used here. Results are exported and execution stops; no winner is
selected automatically and no approval YAML exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment_config import RANDOM_STATE
from config.multiuav_config import (
    FS_OPS,
    MODEL_BENCHMARK_CANDIDATES,
    PRIMARY_TARGET,
    VALIDATION_CASES,
    get_case_feature_set_options,
    get_validation_feature_set,
    validate_primary_ee_feature_list,
)
from src.modeling.validation_common import (
    build_model,
    case_result_dir,
    development_holdout_split,
    evaluate_cv_configuration,
    load_case_dataset,
    parse_model_list,
    save_json,
    summarize_fold_metrics,
)


# Fixed CP1 configurations. Hyperparameter tuning is deliberately reserved for CP2.
CP1_FIXED_PARAMS = {
    "LinearRegression": {},
    "Ridge": {"alpha": 1.0},
    "Lasso": {"alpha": 0.1},
    "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.5},
    "DecisionTree": {},
    "ExtraTrees": {"n_estimators": 100},
    "RandomForest": {"n_estimators": 100},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    "LightGBM": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
    "CatBoost": {"iterations": 100, "learning_rate": 0.1, "depth": 6},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CP1 feature-set screening")
    parser.add_argument("--case", required=True, choices=VALIDATION_CASES)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated screening models or 'all'.",
    )
    args = parser.parse_args()

    case = args.case
    model_names = parse_model_list(args.models, MODEL_BENCHMARK_CANDIDATES)
    df, source_path = load_case_dataset(case)
    dev, holdout, split_manifest = development_holdout_split(df, case)

    out_dir = case_result_dir(case) / "cp1_feature_screen"
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = []
    all_fold_rows = []
    all_oof_rows = []

    print("=" * 78)
    print(f"CP1 FEATURE-SET SCREEN — {case}")
    print(f"Primary target: {PRIMARY_TARGET} [m/Wh]")
    print(f"Source: {source_path}")
    print(f"Development: {len(dev)} | Untouched holdout: {len(holdout)}")
    print("=" * 78)

    for feature_set in get_case_feature_set_options(case):
        features = get_validation_feature_set(feature_set, case)
        validate_primary_ee_feature_list(features)
        missing = [f for f in features if f not in dev.columns]
        if missing:
            raise ValueError(f"{case}/{feature_set} missing features: {missing}")

        print(f"\n[{feature_set}] {len(features)} features")
        model_summaries = []
        for model_name in model_names:
            # Fail early if a dependency/model constructor is invalid.
            build_model(model_name, CP1_FIXED_PARAMS[model_name], RANDOM_STATE)
            fold_df, oof_df = evaluate_cv_configuration(
                dev,
                case,
                features,
                model_name,
                CP1_FIXED_PARAMS[model_name],
            )
            summary = summarize_fold_metrics(fold_df)
            model_summaries.append(summary)

            fold_df.insert(0, "Feature_Set", feature_set)
            oof_df.insert(0, "Feature_Set", feature_set)
            oof_df.insert(1, "Model", model_name)
            all_fold_rows.append(fold_df)
            all_oof_rows.append(oof_df)

            print(
                f"  {model_name:16s} "
                f"R2={summary['CV_R2_mean']:.4f} "
                f"WMAPE={summary['CV_WMAPE_mean']*100:.2f}%"
            )

        # CP1 compares the feature representation, not one cherry-picked model.
        # Aggregate equally across the same fixed model family.
        row = {
            "Case": case,
            "Feature_Set": feature_set,
            "N_Features": len(features),
            "Features": ",".join(features),
            "N_Development": len(dev),
            "N_Holdout_Unused": len(holdout),
            "N_Models": len(model_names),
        }
        for metric in ["R2", "MAE", "RMSE", "WMAPE"]:
            means = [x[f"CV_{metric}_mean"] for x in model_summaries]
            stds = [x[f"CV_{metric}_std"] for x in model_summaries]
            row[f"CV_{metric}_mean_across_models"] = float(pd.Series(means).mean())
            row[f"CV_{metric}_mean_fold_std"] = float(pd.Series(stds).mean())
        evidence_rows.append(row)

    evidence = pd.DataFrame(evidence_rows)
    baseline = evidence[evidence["Feature_Set"] == FS_OPS]
    if baseline.empty:
        raise RuntimeError("FS_OPS baseline is required for CP1 deltas.")
    base_r2 = float(baseline.iloc[0]["CV_R2_mean_across_models"])
    base_wmape = float(baseline.iloc[0]["CV_WMAPE_mean_across_models"])

    evidence["Delta_R2_vs_FS_OPS"] = evidence["CV_R2_mean_across_models"] - base_r2
    # Positive = lower WMAPE than baseline = improvement.
    evidence["Delta_WMAPE_vs_FS_OPS"] = base_wmape - evidence["CV_WMAPE_mean_across_models"]

    fold_all = pd.concat(all_fold_rows, ignore_index=True)
    oof_all = pd.concat(all_oof_rows, ignore_index=True)
    evidence.to_csv(out_dir / "feature_evidence.csv", index=False)
    fold_all.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_all.to_csv(out_dir / "oof_predictions.csv", index=False)

    manifest = {
        "checkpoint": "CP1_FEATURE_SET_SELECTION",
        "case": case,
        "primary_target": PRIMARY_TARGET,
        "source_file": str(source_path),
        "feature_sets_evaluated": get_case_feature_set_options(case),
        "models": model_names,
        "fixed_params": {m: CP1_FIXED_PARAMS[m] for m in model_names},
        "split": split_manifest,
        "holdout_used_for_cp1": False,
        "automatic_winner_selection": False,
    }
    save_json(manifest, out_dir / "run_manifest.json")

    checkpoint_text = f"""CP1 FEATURE SET SELECTION — {case}\n\nFeature evidence has been generated.\nThe untouched holdout has NOT been used.\nNo feature set was selected automatically.\n\nReview:\n  {out_dir / 'feature_evidence.csv'}\n\nThen continue explicitly, for example:\n  python run_model_benchmark.py --case {case} --feature-set FS_OPS_ENV\n"""
    (out_dir / "CHECKPOINT_CP1.txt").write_text(checkpoint_text, encoding="utf-8")

    print("\n" + evidence.to_string(index=False))
    print("\nCP1 COMPLETE — STOPPING FOR FEATURE-SET DECISION")
    print(out_dir / "feature_evidence.csv")


if __name__ == "__main__":
    main()

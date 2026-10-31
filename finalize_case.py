"""Finalize one validation case after explicit CP1 + CP2 decisions.

This stage is invoked only after the researcher explicitly supplies both the
feature set and model family. It then:

1. Reads CP2's best hyperparameters for that chosen model.
2. Fits an evaluation model on the development partition only.
3. Evaluates once on the untouched group holdout.
4. Runs SHAP on the untouched holdout without using it for selection/tuning.
5. Runs group and individual ablation by development GroupKFold.
6. Retrains a deployment artifact on all available rows AFTER evaluation.
7. Writes a reproducible selection manifest (not an approval gate).

Example
-------
python finalize_case.py --case DJI_ONLY --feature-set FS_OPS_ENV --model ExtraTrees
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import (
    ENVIRONMENTAL_FEATURES,
    MODEL_BENCHMARK_CANDIDATES,
    OPERATIONAL_FEATURES,
    PLATFORM_FEATURES,
    PRIMARY_TARGET,
    VALIDATION_CASES,
    get_validation_feature_set,
    validate_primary_ee_feature_list,
)
from config.paths import OUT_BEST_MODEL
from src.modeling.metrics import compute_metrics
from src.modeling.validation_common import (
    TREE_MODEL_NAMES,
    case_result_dir,
    development_holdout_split,
    evaluate_cv_configuration,
    fit_model_bundle,
    load_best_params_for_model,
    load_case_dataset,
    save_json,
    summarize_fold_metrics,
)


def _run_ablation(
    dev: pd.DataFrame,
    case: str,
    full_features: list[str],
    model_name: str,
    params: dict,
    out_dir: Path,
) -> None:
    full_fold, _ = evaluate_cv_configuration(
        dev, case, full_features, model_name, params
    )
    full = summarize_fold_metrics(full_fold)

    groups = {
        "Operational": [f for f in OPERATIONAL_FEATURES if f in full_features],
        "Environmental": [f for f in ENVIRONMENTAL_FEATURES if f in full_features],
        "Platform": [f for f in PLATFORM_FEATURES if f in full_features],
    }

    group_rows = [{
        "Scenario": "Full",
        "Removed": "",
        "N_Features": len(full_features),
        **full,
        "Delta_R2": 0.0,
        "Delta_WMAPE": 0.0,
    }]

    for group_name, removed in groups.items():
        if not removed:
            continue
        kept = [f for f in full_features if f not in removed]
        if not kept:
            continue
        fold_df, _ = evaluate_cv_configuration(dev, case, kept, model_name, params)
        summary = summarize_fold_metrics(fold_df)
        group_rows.append({
            "Scenario": f"Without {group_name}",
            "Removed": ",".join(removed),
            "N_Features": len(kept),
            **summary,
            # Positive means removing the group worsens performance.
            "Delta_R2": full["CV_R2_mean"] - summary["CV_R2_mean"],
            "Delta_WMAPE": summary["CV_WMAPE_mean"] - full["CV_WMAPE_mean"],
        })

    individual_rows = []
    for feature in full_features:
        kept = [f for f in full_features if f != feature]
        if not kept:
            continue
        fold_df, _ = evaluate_cv_configuration(dev, case, kept, model_name, params)
        summary = summarize_fold_metrics(fold_df)
        individual_rows.append({
            "Removed_Feature": feature,
            "N_Features": len(kept),
            **summary,
            "Delta_R2": full["CV_R2_mean"] - summary["CV_R2_mean"],
            "Delta_WMAPE": summary["CV_WMAPE_mean"] - full["CV_WMAPE_mean"],
        })

    pd.DataFrame(group_rows).to_csv(out_dir / "group_ablation.csv", index=False)
    pd.DataFrame(individual_rows).sort_values(
        ["Delta_R2", "Delta_WMAPE"], ascending=[False, False]
    ).to_csv(out_dir / "individual_ablation.csv", index=False)


def _run_shap(
    bundle,
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Compute SHAP on holdout. Return status metadata instead of hiding failure."""
    shap_dir = out_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    features = bundle.features

    X_hold_t = bundle.transform(holdout[features])
    if isinstance(X_hold_t, np.ndarray):
        X_hold_plot = pd.DataFrame(X_hold_t, columns=features, index=holdout.index)
    else:
        X_hold_plot = X_hold_t.copy()

    # Bound generic explainers to keep a finalized case practical.
    max_rows = min(200, len(X_hold_plot))
    X_explain = X_hold_plot.iloc[:max_rows].copy()

    try:
        if bundle.model_name in TREE_MODEL_NAMES:
            explainer = shap.TreeExplainer(bundle.model)
            shap_values = explainer.shap_values(X_explain)
        else:
            X_dev_t = bundle.transform(dev[features])
            if isinstance(X_dev_t, np.ndarray):
                X_dev_t = pd.DataFrame(X_dev_t, columns=features, index=dev.index)
            background = shap.sample(X_dev_t, min(100, len(X_dev_t)), random_state=42)
            explainer = shap.Explainer(bundle.model.predict, background)
            explanation = explainer(X_explain)
            shap_values = explanation.values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        if shap_values.shape[1] != len(features):
            raise ValueError(
                f"Unexpected SHAP shape {shap_values.shape} for {len(features)} features"
            )

        importance = pd.DataFrame({
            "Feature": features,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(shap_dir / "shap_importance.csv", index=False)

        shap.summary_plot(shap_values, X_explain, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_bar.png", dpi=160, bbox_inches="tight")
        plt.close()

        return {
            "status": "ok",
            "n_explained": int(len(X_explain)),
            "importance_file": str(shap_dir / "shap_importance.csv"),
        }
    except Exception as exc:
        # Finalization remains usable, but the failure is explicit and auditable.
        err = f"{type(exc).__name__}: {exc}"
        (shap_dir / "SHAP_ERROR.txt").write_text(err, encoding="utf-8")
        return {"status": "error", "error": err}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize one validation case")
    parser.add_argument("--case", required=True, choices=VALIDATION_CASES)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--model", required=True, choices=MODEL_BENCHMARK_CANDIDATES)
    args = parser.parse_args()

    case = args.case
    features = get_validation_feature_set(args.feature_set, case)
    validate_primary_ee_feature_list(features)

    cp2_path = case_result_dir(case) / "cp2_model_benchmark" / "model_evidence.csv"
    if not cp2_path.exists():
        raise FileNotFoundError(
            f"Missing CP2 evidence: {cp2_path}\n"
            f"Run run_model_benchmark.py for {case}/{args.feature_set} first."
        )
    cp2_df = pd.read_csv(cp2_path)
    if "Feature_Set" in cp2_df.columns:
        feature_sets = set(cp2_df["Feature_Set"].astype(str))
        if args.feature_set not in feature_sets:
            raise ValueError(
                f"CP2 evidence was not produced for feature set {args.feature_set}. "
                f"Found: {sorted(feature_sets)}"
            )

    params = load_best_params_for_model(cp2_path, args.model)
    df, source_path = load_case_dataset(case)
    dev, holdout, split_manifest = development_holdout_split(df, case)

    out_dir = case_result_dir(case) / "finalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(OUT_BEST_MODEL) / "validation" / case.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"FINALIZE CASE — {case}")
    print(f"Feature set: {args.feature_set}")
    print(f"Model: {args.model}")
    print(f"Best CP2 params: {params}")
    print("=" * 78)

    evaluation_bundle = fit_model_bundle(
        args.model,
        params,
        dev[features],
        dev[PRIMARY_TARGET].to_numpy(float),
        features,
    )
    holdout_pred = evaluation_bundle.predict(holdout[features])
    metrics = compute_metrics(holdout[PRIMARY_TARGET].to_numpy(float), holdout_pred)

    pred_df = holdout[["flight_uid", "dataset_id", "date", PRIMARY_TARGET]].copy()
    pred_df["predicted_energy_efficiency"] = holdout_pred
    pred_df["residual"] = pred_df[PRIMARY_TARGET] - pred_df["predicted_energy_efficiency"]
    pred_df.to_csv(out_dir / "holdout_predictions.csv", index=False)

    evaluation_row = {
        "Case": case,
        "Feature_Set": args.feature_set,
        "Model": args.model,
        "Best_Params_JSON": json.dumps(params, sort_keys=True),
        "N_Features": len(features),
        "N_Development": len(dev),
        "N_Holdout": len(holdout),
        **metrics,
    }
    pd.DataFrame([evaluation_row]).to_csv(out_dir / "holdout_metrics.csv", index=False)

    # Save the exact model used for untouched holdout evaluation.
    joblib.dump(evaluation_bundle, model_dir / "evaluation_model.pkl")

    print("Running SHAP on untouched holdout...")
    shap_status = _run_shap(evaluation_bundle, dev, holdout, out_dir)

    print("Running development-CV ablation...")
    _run_ablation(dev, case, features, args.model, params, out_dir)

    # Deployment artifact is retrained only AFTER all holdout metrics were fixed.
    deployment_bundle = fit_model_bundle(
        args.model,
        params,
        df[features],
        df[PRIMARY_TARGET].to_numpy(float),
        features,
    )
    joblib.dump(deployment_bundle, model_dir / "final_model.pkl")

    manifest = {
        "checkpoint_source": "explicit CLI decision after CP1/CP2 evidence",
        "case": case,
        "feature_set": args.feature_set,
        "features": features,
        "model": args.model,
        "best_params": params,
        "primary_target": PRIMARY_TARGET,
        "target_unit": "m/Wh",
        "source_file": str(source_path),
        "split": split_manifest,
        "holdout_metrics": metrics,
        "shap": shap_status,
        "evaluation_model": str(model_dir / "evaluation_model.pkl"),
        "deployment_model": str(model_dir / "final_model.pkl"),
        "note": (
            "evaluation_model was trained on development only; final_model was retrained "
            "on all rows only after untouched-holdout evaluation was completed."
        ),
    }
    save_json(manifest, out_dir / "selection_manifest.json")
    save_json(manifest, model_dir / "model_manifest.json")

    print("\nUntouched holdout metrics:")
    for key in ["R2", "MAE", "RMSE", "WMAPE", "Bias"]:
        val = metrics.get(key)
        if key == "WMAPE" and np.isfinite(val):
            print(f"  {key}: {val*100:.3f}%")
        else:
            print(f"  {key}: {val:.6f}" if np.isfinite(val) else f"  {key}: nan")
    print(f"\nSaved: {out_dir}")
    print("Case finalized. Final data-setting selection remains CP3.")


if __name__ == "__main__":
    main()

"""CP3 evidence builder: compare finalized DJI_ONLY, VTOL_ONLY, COMBINED cases.

This script never selects a winner. It gathers untouched-holdout metrics from
all finalized cases into one comparable table and stops for an explicit
validated-configuration decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import VALIDATION_CASES
from config.paths import OUT_RES_MULTIUAV
from src.modeling.validation_common import case_result_dir, save_json


def main() -> None:
    rows = []
    missing = []
    for case in VALIDATION_CASES:
        metrics_path = case_result_dir(case) / "finalized" / "holdout_metrics.csv"
        manifest_path = case_result_dir(case) / "finalized" / "selection_manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            missing.append(case)
            continue
        row = pd.read_csv(metrics_path).iloc[0].to_dict()
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        row["SHAP_Status"] = manifest.get("shap", {}).get("status", "unknown")
        row["Target"] = manifest.get("primary_target", "energy_efficiency")
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            "No finalized validation cases found. Run finalize_case.py first."
        )

    out_dir = Path(OUT_RES_MULTIUAV) / "validation" / "cp3_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    preferred_cols = [
        "Case", "Feature_Set", "Model", "Target", "N_Features",
        "N_Development", "N_Holdout", "R2", "MAE", "RMSE", "WMAPE",
        "Bias", "MedAE", "SHAP_Status", "Best_Params_JSON",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    df = df[cols]
    df.to_csv(out_dir / "validation_case_evidence.csv", index=False)

    save_json(
        {
            "checkpoint": "CP3_VALIDATED_CONFIGURATION_SELECTION",
            "cases_available": df["Case"].astype(str).tolist(),
            "cases_missing": missing,
            "automatic_winner_selection": False,
            "instruction": (
                "Review predictive accuracy, WMAPE, stability/SHAP/ablation evidence, "
                "and research scope. Then record the selected case explicitly with "
                "select_validated_configuration.py --case <CASE>."
            ),
        },
        out_dir / "run_manifest.json",
    )
    (out_dir / "CHECKPOINT_CP3.txt").write_text(
        "CP3 VALIDATED CONFIGURATION SELECTION\n\n"
        "No data/model setting has been selected automatically.\n"
        "Review validation_case_evidence.csv plus each case's SHAP/ablation outputs.\n\n"
        "Continue explicitly, e.g.:\n"
        "  python select_validated_configuration.py --case COMBINED\n",
        encoding="utf-8",
    )

    print(df.to_string(index=False))
    if missing:
        print(f"\nNot yet finalized: {missing}")
    print("\nCP3 evidence complete — STOPPING FOR EXPLICIT CONFIGURATION DECISION")


if __name__ == "__main__":
    main()

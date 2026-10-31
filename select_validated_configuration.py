"""Record the explicit CP3 validated-configuration decision.

This is not an approval gate. The command itself is the decision. The script
copies no metrics and selects nothing automatically; it records exactly which
already-finalized case the researcher chose and where its model artifacts live.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import VALIDATION_CASES
from config.paths import OUT_RES_MULTIUAV, OUT_BEST_MODEL
from src.modeling.validation_common import case_result_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record CP3 selected validation case")
    parser.add_argument("--case", required=True, choices=VALIDATION_CASES)
    args = parser.parse_args()

    manifest_path = case_result_dir(args.case) / "finalized" / "selection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Case {args.case} is not finalized: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        case_manifest = json.load(f)

    selected = {
        "selected_case": args.case,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": case_manifest["feature_set"],
        "features": case_manifest["features"],
        "model": case_manifest["model"],
        "best_params": case_manifest["best_params"],
        "primary_target": case_manifest["primary_target"],
        "holdout_metrics": case_manifest["holdout_metrics"],
        "model_path": str(
            Path(OUT_BEST_MODEL) / "validation" / args.case.lower() / "final_model.pkl"
        ),
        "source_manifest": str(manifest_path),
        "decision_mechanism": "explicit CLI checkpoint selection; no approved flag",
    }

    out_dir = Path(OUT_RES_MULTIUAV) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_configuration.json"
    save_json(selected, out_path)
    print(f"Recorded CP3 selection: {args.case}")
    print(f"Saved: {out_path}")
    print("Next: build/refit the Multi-Hub segment Energy model, then run final Multi-Hub EE analysis.")


if __name__ == "__main__":
    main()

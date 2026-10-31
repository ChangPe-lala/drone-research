"""Supporting Multi-UAV/data-extension experiments without final approval gates.

D00-D06 prepare/audit VTOL + combined data. D07-D12 are supporting Multi-UAV
experiments (within-domain diagnostics, transfer, domain shift, universal and
platform-aware learning). They are NOT the canonical CP1/CP2/CP3 validation
runners and they do not select/freeze the final research configuration.

D13 approval-YAML final selection is intentionally removed from this runner.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = [
    ("D00", "VTOL Raw Audit", "src/data_processing/vtol_bag_audit.py"),
    ("D01", "VTOL Raw Extraction + Weather", "src/data_processing/build_vtol_raw_dataset.py"),
    ("D02", "VTOL Cleaning", "src/data_processing/vtol_data_cleaning.py"),
    ("D03", "VTOL Flight-Level Features", "src/data_processing/vtol_feature_engineering.py"),
    ("D04", "DJI Flight-Level Features", "src/data_processing/feature_engineering.py"),
    ("D05", "Explicit DJI+VTOL Harmonization", "src/data_processing/harmonize_uav_datasets.py"),
    ("D06", "Harmonized Data Validation", "src/data_processing/validate_multiuav_data.py"),
    ("D06A", "Target Integration Audit", "src/analysis/target_integration_audit.py"),
    ("D07", "Legacy Within-Domain Diagnostic", "src/modeling/01_within_domain.py"),
    ("D07A", "Aero Feature Supporting Ablation", "src/modeling/01c_aero_feature_ablation.py"),
    ("D08", "Cross-Domain Transfer", "src/modeling/02_cross_domain.py"),
    ("D09", "Target Strategy Supporting Experiment", "src/modeling/03_target_strategy.py"),
    ("D10", "Domain Shift", "src/modeling/04_domain_shift_analysis.py"),
    ("D11", "Universal Combined Training", "src/modeling/05_combined_training.py"),
    ("D12", "Platform-Aware Supporting Experiment", "src/modeling/06_platform_aware.py"),
]


def run(stage):
    sid, name, script = stage
    path = PROJECT_ROOT / script
    print(f"\n{'='*70}\n[{sid}] {name}\n{script}\n{'='*70}")
    env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run([sys.executable, str(path)], check=True, cwd=PROJECT_ROOT, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--from-stage", default=None)
    args = parser.parse_args()
    ids = [s[0] for s in STAGES]
    if args.from_stage:
        if args.from_stage not in ids: parser.error(f"Unknown stage {args.from_stage}")
        chosen = STAGES[ids.index(args.from_stage):]
    elif args.stage == "all":
        chosen = STAGES
    else:
        if args.stage not in ids: parser.error(f"Unknown stage {args.stage}")
        chosen = [STAGES[ids.index(args.stage)]]
    for stage in chosen: run(stage)
    print("\nSupporting extension stages complete. No final configuration was auto-selected.")

if __name__ == "__main__":
    main()

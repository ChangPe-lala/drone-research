"""Canonical checkpoint-based RS pipeline dispatcher.

There is intentionally NO ``all`` mode because CP1/CP2/CP3 require explicit
research decisions. Each stage runs deterministically, exports evidence, and
stops at the next checkpoint.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script)] + extra
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _status() -> None:
    artifacts = {
        "DJI CP1": "output/results/validation/dji_only/cp1_feature_screen/feature_evidence.csv",
        "VTOL CP1": "output/results/validation/vtol_only/cp1_feature_screen/feature_evidence.csv",
        "COMBINED CP1": "output/results/validation/combined/cp1_feature_screen/feature_evidence.csv",
        "DJI finalized": "output/results/validation/dji_only/finalized/holdout_metrics.csv",
        "VTOL finalized": "output/results/validation/vtol_only/finalized/holdout_metrics.csv",
        "COMBINED finalized": "output/results/validation/combined/finalized/holdout_metrics.csv",
        "CP3 evidence": "output/results/validation/cp3_comparison/validation_case_evidence.csv",
        "CP3 selected": "output/results/validation/selected_configuration.json",
        "Segment evidence": "output/results/multihub/segment_model_checkpoint/segment_model_evidence.csv",
        "Segment model": "output/models/multihub/segment_energy_model.pkl",
        "Final RQ5": "output/results/multihub/final_rq5/mission_recommendations.csv",
    }
    print("CHECKPOINT PIPELINE STATUS")
    for label, rel in artifacts.items():
        path = PROJECT_ROOT / rel
        print(f"  {'DONE' if path.exists() else '----'}  {label:20s} {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RS checkpoint pipeline; no approval YAML")
    parser.add_argument("--stage", required=True, choices=[
        "status", "cp1", "cp2", "finalize", "cp3", "select",
        "multihub-build", "segment-benchmark", "segment-fit", "multihub-final",
    ])
    parser.add_argument("--case", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", default="all")
    args = parser.parse_args()

    if args.stage == "status":
        _status(); return
    if args.stage == "cp1":
        if not args.case: parser.error("cp1 requires --case")
        _run("run_feature_screen.py", ["--case", args.case, "--models", args.models]); return
    if args.stage == "cp2":
        if not args.case or not args.feature_set: parser.error("cp2 requires --case and --feature-set")
        _run("run_model_benchmark.py", ["--case", args.case, "--feature-set", args.feature_set, "--models", args.models]); return
    if args.stage == "finalize":
        if not args.case or not args.feature_set or not args.model:
            parser.error("finalize requires --case, --feature-set and --model")
        _run("finalize_case.py", ["--case", args.case, "--feature-set", args.feature_set, "--model", args.model]); return
    if args.stage == "cp3":
        _run("compare_validation_cases.py", []); return
    if args.stage == "select":
        if not args.case: parser.error("select requires --case")
        _run("select_validated_configuration.py", ["--case", args.case]); return
    if args.stage == "multihub-build":
        extra = ["--models", args.models]
        if args.case: extra += ["--case", args.case]
        _run("run_multihub_g00_g07.py", extra); return
    if args.stage == "segment-benchmark":
        _run("benchmark_segment_energy_models.py", ["--models", args.models]); return
    if args.stage == "segment-fit":
        if not args.model: parser.error("segment-fit requires --model")
        _run("fit_segment_energy_model.py", ["--model", args.model]); return
    if args.stage == "multihub-final":
        extra = ["--case", args.case] if args.case else []
        _run("run_final_multihub.py", extra); return


if __name__ == "__main__":
    main()

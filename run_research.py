
"""Unified entry point for the research pipeline.

This file only dispatches existing research scripts.
Scientific implementations and research decisions are unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CASES = ("DJI_ONLY", "VTOL_ONLY", "COMBINED")

SUPPORTING_STAGES = (
    "D07", "D07A", "D08", "D09",
    "D10", "D11", "D12",
)

VTOL_STAGES = {
    "vtol-audit": "D00",
    "vtol-extract": "D01",
    "vtol-clean": "D02",
    "vtol-features": "D03",
}


def execute(arguments: list[str], *, dry_run: bool = False) -> None:
    """Execute a command from the project root."""

    command = [sys.executable, *arguments]
    print("\nRUN:", " ".join(command), flush=True)

    if not dry_run:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )


def script(name: str, *args: str, dry_run: bool = False) -> None:
    """Execute an existing root-level research script."""

    path = ROOT / name

    if not path.is_file():
        raise FileNotFoundError(f"Missing script: {path}")

    execute([str(path), *args], dry_run=dry_run)


def module(name: str, *, dry_run: bool = False) -> None:
    """Run a Python module with the correct project import path."""

    execute(["-m", name], dry_run=dry_run)


def extension(stage: str, *, dry_run: bool = False) -> None:
    """Run one existing dataset-extension stage."""

    script(
        "run_dataset_extension.py",
        "--stage",
        stage,
        dry_run=dry_run,
    )


def required(parser, value, argument: str) -> str:
    if not value:
        parser.error(f"{argument} is required")
    return value


def show_status() -> None:
    """Report existing artifacts without modifying them."""

    artifacts = {
        "DJI cleaned":
            "data/processed/cleaned_flight_data.csv",
        "VTOL cleaned":
            "data/processed/vtol_cleaned_flight_data.csv",
        "DJI features":
            "output/data/features/flight_level_features.csv",
        "VTOL features":
            "output/data/features/vtol_flight_level_features.csv",
        "Selected configuration":
            "output/results/multiuav/validation/"
            "selected_configuration.json",
        "Segment model":
            "output/models/multihub/segment_energy_model.pkl",
        "Routing comparison":
            "output/results/multihub/"
            "route_baseline_comparison.csv",
        "Routing summary":
            "output/results/multihub/"
            "route_baseline_summary.csv",
    }

    print("\nRESEARCH PIPELINE STATUS\n")

    for label, relative_path in artifacts.items():
        path = ROOT / relative_path
        status = "FOUND" if path.is_file() else "MISSING"
        print(f"{status:8} {label:25} {relative_path}")


def main() -> None:
    # Preserve the original run_research.py --stage ... interface.
    if len(sys.argv) > 1 and sys.argv[1] == "--stage":
        from run_pipeline import main as pipeline_main
        pipeline_main()
        return

    parser = argparse.ArgumentParser(
        description="Drone energy and routing research pipeline"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing research stages",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    data = commands.add_parser("data")
    data.add_argument(
        "--stage",
        required=True,
        choices=(
            "dji-extract",
            "dji-audit",
            "dji-clean",
            "dji-features",
            "dji-validate",
            "vtol-audit",
            "vtol-extract",
            "vtol-clean",
            "vtol-features",
            "vtol-all",
            "harmonize",
        ),
    )

    supporting = commands.add_parser("supporting")
    supporting.add_argument(
        "--stage",
        default="all",
        choices=("all", *SUPPORTING_STAGES),
    )

    validation = commands.add_parser("validation")
    validation.add_argument(
        "--stage",
        required=True,
        choices=(
            "cp1",
            "cp2",
            "finalize",
            "cp3",
            "select",
            "ablation",
        ),
    )
    validation.add_argument("--case", choices=CASES)
    validation.add_argument("--feature-set")
    validation.add_argument("--model")
    validation.add_argument("--models", default="all")

    multihub = commands.add_parser("multihub")
    multihub.add_argument(
        "--stage",
        required=True,
        choices=(
            "segment",
            "fit",
            "routing",
            "sensitivity",
        ),
    )
    multihub.add_argument("--case", choices=CASES)
    multihub.add_argument("--model")
    multihub.add_argument("--models", default="all")

    audit = commands.add_parser("audit")
    audit.add_argument(
        "--stage",
        required=True,
        choices=("paper", "sanity"),
    )

    commands.add_parser("figures")
    commands.add_parser("status")

    args = parser.parse_args()
    dry = args.dry_run

    # ----------------------------------------------------
    # DATA PROCESSING
    # ----------------------------------------------------
    if args.command == "data":
        stage = args.stage

        dji_modules = {
            "dji-extract":
                "src.data_processing.build_raw_dataset",
            "dji-clean":
                "src.data_processing.apply_data_cleaning",
            "dji-validate":
                "src.data_processing.validate_processed_data",
        }

        if stage in dji_modules:
            module(dji_modules[stage], dry_run=dry)

        elif stage == "dji-audit":
            for name in (
                "src.data_processing.validate_raw_data",
                "src.data_processing.detect_data_issues",
            ):
                module(name, dry_run=dry)

        elif stage == "dji-features":
            extension("D04", dry_run=dry)

        elif stage in VTOL_STAGES:
            extension(VTOL_STAGES[stage], dry_run=dry)

        elif stage == "vtol-all":
            for step in ("D00", "D01", "D02", "D03"):
                extension(step, dry_run=dry)

        elif stage == "harmonize":
            for step in ("D05", "D06", "D06A"):
                extension(step, dry_run=dry)

    # ----------------------------------------------------
    # SUPPORTING EXPERIMENTS
    # ----------------------------------------------------
    elif args.command == "supporting":
        if args.stage == "all":
            script(
                "run_dataset_extension.py",
                "--from-stage",
                "D07",
                dry_run=dry,
            )
        else:
            extension(args.stage, dry_run=dry)

    # ----------------------------------------------------
    # CP1 - CP3 VALIDATION
    # ----------------------------------------------------
    elif args.command == "validation":
        stage = args.stage

        if stage == "ablation":
            script("run_full_ablation.py", dry_run=dry)
            return

        options = ["--stage", stage]

        if stage in ("cp1", "cp2", "finalize", "select"):
            options += [
                "--case",
                required(parser, args.case, "--case"),
            ]

        if stage in ("cp2", "finalize"):
            options += [
                "--feature-set",
                required(
                    parser,
                    args.feature_set,
                    "--feature-set",
                ),
            ]

        if stage == "finalize":
            options += [
                "--model",
                required(parser, args.model, "--model"),
            ]

        if stage in ("cp1", "cp2"):
            options += ["--models", args.models]

        script("run_pipeline.py", *options, dry_run=dry)

    # ----------------------------------------------------
    # MULTI-HUB RESEARCH
    # ----------------------------------------------------
    elif args.command == "multihub":
        stage = args.stage

        if stage == "segment":
            options = ["--models", args.models]

            if args.case:
                options += ["--case", args.case]

            script(
                "run_multihub_g00_g07.py",
                *options,
                dry_run=dry,
            )

        elif stage == "fit":
            model = required(parser, args.model, "--model")

            script(
                "fit_segment_energy_model.py",
                "--model",
                model,
                dry_run=dry,
            )

        elif stage == "routing":
            options = (
                ["--case", args.case]
                if args.case
                else []
            )

            script(
                "run_final_multihub.py",
                *options,
                dry_run=dry,
            )

        elif stage == "sensitivity":
            options = (
                ["--case", args.case]
                if args.case
                else []
            )

            script(
                "run_multihub_sensitivity.py",
                *options,
                dry_run=dry,
            )

    # ----------------------------------------------------
    # AUDITS AND FIGURES
    # ----------------------------------------------------
    elif args.command == "audit":
        audit_scripts = {
            "paper": "run_paper_number_audit.py",
            "sanity": "run_g14_sanity_audit.py",
        }

        script(audit_scripts[args.stage], dry_run=dry)

    elif args.command == "figures":
        script("generate_result_figures.py", dry_run=dry)

    elif args.command == "status":
        show_status()


if __name__ == "__main__":
    main()

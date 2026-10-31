
"""Run an isolated Multi-Hub topology sensitivity experiment.

Each configuration runs in a separate process using the
same frozen segment energy model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multihub_config import (
    HUB_DISTANCE_THRESHOLD_M,
    ON_ROUTE_HUB_INTERVAL_M,
    ROUTING_POLICY_SCHEMA_VERSION,
)
from config.multiuav_config import VALIDATION_CASES
from config.paths import OUTPUT_DIR

from src.pipelines.multihub import (
    prepare_segment_data,
    run_routing_pipeline,
)


def save_variant_manifest(case: str, model_path: Path) -> None:
    """Save the configuration used for this sensitivity run."""

    result_dir = Path(OUTPUT_DIR) / "results" / "multihub"
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "validation_case": case,
        "anchor_interval_m": ON_ROUTE_HUB_INTERVAL_M,
        "cluster_threshold_m": HUB_DISTANCE_THRESHOLD_M,
        "output_dir": str(OUTPUT_DIR),
        "frozen_segment_model": str(model_path.resolve()),
        "routing_policy_schema_version": ROUTING_POLICY_SCHEMA_VERSION,
    }

    manifest_path = result_dir / "variant_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one isolated Multi-Hub sensitivity configuration"
    )

    parser.add_argument(
        "--case",
        choices=VALIDATION_CASES,
        required=True,
        help="Validation case used for this configuration",
    )

    args = parser.parse_args()

    # The segment model must already be trained and frozen.
    model_value = os.environ.get("RS_SEGMENT_MODEL_PATH")

    if not model_value:
        raise RuntimeError(
            "RS_SEGMENT_MODEL_PATH is required. "
            "A frozen segment model must be provided."
        )

    model_path = Path(model_value)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen segment model not found: {model_path}"
        )

    print("=" * 60)
    print("MULTI-HUB SENSITIVITY EXPERIMENT")
    print(f"Validation case: {args.case}")
    print(f"Anchor interval: {ON_ROUTE_HUB_INTERVAL_M} m")
    print(f"Cluster threshold: {HUB_DISTANCE_THRESHOLD_M} m")
    print(f"Frozen model: {model_path}")
    print("=" * 60)

    # G00-G06: Rebuild segment data for this topology.
    # Keep the original variant's validation behavior.
    prepare_segment_data(
        args.case,
        check_nonempty=False,
    )

    # G08-G14: Execute routing with the shared workflow.
    # Also produce RQ5, strategy comparisons and topology audit.
    run_routing_pipeline(
        case=args.case,
        include_graph=True,
        include_final_reports=True,
        include_topology_audit=True,
    )

    # Preserve the manifest expected by sensitivity aggregation.
    save_variant_manifest(args.case, model_path)

    print("=" * 60)
    print("SENSITIVITY CONFIGURATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

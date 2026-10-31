
"""Run segment preparation and model benchmarking (G00-G07)."""

from __future__ import annotations

import argparse

from config.multiuav_config import VALIDATION_CASES
from src.multihub.segment_modeling import benchmark_segment_models
from src.pipelines.multihub import prepare_segment_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Hub G00-G07 checkpoint runner"
    )

    parser.add_argument(
        "--case",
        choices=VALIDATION_CASES,
        default=None,
        help="Override the selected validation case",
    )

    parser.add_argument(
        "--models",
        default="all",
        help="Segment models to benchmark",
    )

    args = parser.parse_args()

    # G00-G06: Prepare and validate segment data.
    prepare_segment_data(args.case)

    # G07: Benchmark models without automatic selection.
    benchmark_segment_models(args.models)

    print("\nG00-G07 evidence complete.")
    print("STOP: Review benchmark results and select a segment model.")
    print("Then run:")
    print("  python fit_segment_energy_model.py --model MODEL_NAME")


if __name__ == "__main__":
    main()

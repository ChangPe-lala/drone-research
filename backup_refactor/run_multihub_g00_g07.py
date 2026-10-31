"""Run G00-G06, validate segments, then benchmark G07 models and STOP.

The old segment-model approval YAML is removed from the canonical flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.multiuav_config import VALIDATION_CASES
from src.multihub.build_segment_features import build_segment_features
from src.multihub.build_trajectory_index import build_trajectory_index
from src.multihub.cluster_hub_nodes import cluster_hub_nodes
from src.multihub.detect_hub_nodes import detect_hub_nodes
from src.multihub.load_multihub_data import load_and_normalize_data
from src.multihub.map_trajectory_nodes import map_trajectory_nodes
from src.multihub.segment_modeling import benchmark_segment_models
from src.multihub.segment_trajectories import segment_trajectories
from src.multihub.validate_segment_dataset import validate_segment_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Hub G00-G07 checkpoint runner")
    parser.add_argument("--case", choices=VALIDATION_CASES, default=None,
                        help="Override CP3 selected case")
    parser.add_argument("--models", default="all", help="Segment models to benchmark")
    args = parser.parse_args()

    df = load_and_normalize_data(args.case)          # G00
    build_trajectory_index(df)                       # G01
    candidates = detect_hub_nodes(df)                # G02
    nodes = cluster_hub_nodes(candidates)             # G03
    mapping = map_trajectory_nodes(df, nodes)         # G04
    segments = segment_trajectories(df, mapping)      # G05
    if segments.empty:
        raise RuntimeError("G05 produced no valid segments.")
    features = build_segment_features(segments)       # G06
    validate_segment_dataset(features)                # pre-G07 audit
    benchmark_segment_models(args.models)             # G07 evidence only

    print("\nG00-G07 evidence complete.")
    print("STOP: choose the segment Energy model, then run:")
    print("  python fit_segment_energy_model.py --model ExtraTrees")


if __name__ == "__main__":
    main()


"""Run the complete Multi-Hub routing evaluation (G08-G14)."""

from __future__ import annotations

import argparse

from config.multiuav_config import VALIDATION_CASES
from src.pipelines.multihub import run_routing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Hub routing and final evaluation"
    )

    parser.add_argument(
        "--case",
        choices=VALIDATION_CASES,
        default=None,
        help="Override the selected validation case",
    )

    args = parser.parse_args()

    # G08-G14: Build graph, predict energy and evaluate routes.
    # Final RQ5: Compare historical and recommended routes.
    # Topology: Audit graph structure and hub participation.
    run_routing_pipeline(
        case=args.case,
        include_graph=True,
        include_final_reports=True,
        include_topology_audit=True,
    )

    print("\nMulti-Hub routing evaluation complete.")


if __name__ == "__main__":
    main()


"""Run Multi-Hub routing stages G09-G14 using an existing graph."""

from __future__ import annotations

from src.pipelines.multihub import run_routing_pipeline


def main() -> None:
    print("=" * 60)
    print("STARTING ROUTING PIPELINE (G09-G14)")
    print("=" * 60)

    # Reuse the existing graph produced by G08.
    # Run G09-G14 and the standard policy comparison.
    run_routing_pipeline(
        include_graph=False,
        include_final_reports=False,
        include_topology_audit=False,
    )

    print("=" * 60)
    print("ROUTING PIPELINE (G09-G14) COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

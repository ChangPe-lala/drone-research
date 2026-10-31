from __future__ import annotations
import argparse
from src.multihub.segment_modeling import benchmark_segment_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark segment Energy models and stop at checkpoint")
    parser.add_argument("--models", default="all", help="Comma-separated models or all")
    args = parser.parse_args()
    benchmark_segment_models(args.models)

from __future__ import annotations
import argparse
from config.multiuav_config import MODEL_BENCHMARK_CANDIDATES
from src.multihub.segment_modeling import fit_selected_segment_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit explicitly selected segment Energy model")
    parser.add_argument("--model", required=True, choices=MODEL_BENCHMARK_CANDIDATES)
    args = parser.parse_args()
    fit_selected_segment_model(args.model)

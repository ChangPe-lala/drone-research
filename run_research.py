"""Compatibility entry point for the new checkpoint pipeline.

The former YAML human-gate orchestrator has been retired. Use the same
arguments documented by run_pipeline.py.
"""
from run_pipeline import main

if __name__ == "__main__":
    main()

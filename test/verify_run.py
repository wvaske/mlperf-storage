#!/usr/bin/env python3
"""
Standalone script to process a benchmark result directory and create BenchmarkResult and BenchmarkRun objects.
"""

import argparse
import json
import pdb
import sys
import os
from pathlib import Path

# Add the project root to the Python path so we can import mlpstorage modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlpstorage.rules import BenchmarkResult, BenchmarkRun
from mlpstorage.mlps_logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Process a benchmark result directory and create BenchmarkResult and BenchmarkRun objects"
    )
    parser.add_argument(
        "result_dir",
        help="Path to the benchmark result directory containing metadata and summary files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file to write the result dictionary (default: stdout)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "verbose" if args.verbose else "info"
    logger = setup_logging(name='process_benchmark_result', stream_log_level=log_level)

    # Validate the result directory exists
    if not os.path.exists(args.result_dir):
        logger.error(f"Result directory does not exist: {args.result_dir}")
        sys.exit(1)

    if not os.path.isdir(args.result_dir):
        logger.error(f"Path is not a directory: {args.result_dir}")
        sys.exit(1)

    try:
        # Create BenchmarkResult object
        logger.info(f"Processing benchmark result directory: {args.result_dir}")
        benchmark_result = BenchmarkResult(args.result_dir, logger)

        logger.verbose(f'BenchmarkResult __dict__:')
        print(json.dumps(benchmark_result.__dict__, indent=2, default=str))

        # Create BenchmarkRun object
        logger.info("Creating BenchmarkRun object from BenchmarkResult")
        benchmark_run = BenchmarkRun(benchmark_result=benchmark_result, logger=logger)

        # Convert to dictionary
        result_dict = benchmark_run.as_dict()
        output_json = json.dumps(result_dict, indent=2, default=str)

        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            logger.info(f"Result written to: {args.output}")
        else:
            print(output_json)

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing benchmark result: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
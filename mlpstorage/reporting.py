import csv
import json
import os.path
import pprint
import sys

from dataclasses import dataclass
from typing import List, Dict, Any

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, MODELS, ACCELERATORS
from mlpstorage.rules import get_runs_files, BenchmarkVerifier, BenchmarkRun, Issue
from mlpstorage.utils import flatten_nested_dict, remove_nan_values

@dataclass
class Result:
    multi: bool
    benchmark_type: BENCHMARK_TYPES
    benchmark_command: str
    benchmark_model: [LLM_MODELS, MODELS]
    benchmark_run: BenchmarkRun
    issues: List[Issue]
    category: PARAM_VALIDATION
    metrics: Dict[str, Any]


class ReportGenerator:

    def __init__(self, results_dir, args=None, logger=None):
        self.args = args
        if self.args is not None:
            self.debug = self.args.debug or MLPS_DEBUG
        else:
            self.debug = MLPS_DEBUG

        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name=f"mlpstorage_reporter")
            apply_logging_options(self.logger, args)

        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            self.logger.error(f'Results directory {self.results_dir} does not exist')
            sys.exit(EXIT_CODE.FILE_NOT_FOUND)

        self.run_results = dict()           # {run_id : result_dict }
        self.workload_results = dict()      # {(model) | (model, accelerator) : result_dict }
        self.accumulate_results()
        self.print_results()

    def generate_reports(self):
        # Verify the results directory exists:
        self.logger.info(f'Generating reports for {self.results_dir}')
        run_result_dicts = [report.benchmark_run.as_dict() for report in self.run_results.values()]

        self.write_csv_file(run_result_dicts)
        self.write_json_file(run_result_dicts)
            
        return EXIT_CODE.SUCCESS

    def accumulate_results(self):
        """
        This function will look through the result_files and generate a result dictionary for each run by reading the metadata.json and summary.json files.

        If the metadata.json file does not exist, log an error and continue
        If summary.json files does not exist, set status=Failed and only use data from metadata.json the run_info from the result_files dictionary
        :return:
        """
        benchmark_runs = get_runs_files(self.results_dir, logger=self.logger)

        self.logger.info(f'Accumulating results from {len(benchmark_runs)} runs')
        for benchmark_run in benchmark_runs:
            self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')
            verifier = BenchmarkVerifier(benchmark_run, logger=self.logger)
            category = verifier.verify()
            issues = verifier.issues
            result_dict = dict(
                multi=False,
                benchmark_run=benchmark_run,
                benchmark_type=benchmark_run.benchmark_type,
                benchmark_command=benchmark_run.command,
                benchmark_model=benchmark_run.model,
                issues=issues,
                category=category,
                metrics=benchmark_run.metrics
            )
            self.run_results[benchmark_run.run_id] = Result(**result_dict)

        # Group runs for workload to run additional verifiers
        # These will be manually defined as these checks align with a specific submission version
        # I need to group by model. For training workloads we also group by accelerator but the same checker
        # is used based on model.
        workload_runs = dict()

        for benchmark_run in benchmark_runs:
            workload_key = (benchmark_run.model, benchmark_run.accelerator)
            if workload_key not in workload_runs.keys():
                workload_runs[workload_key] = []
            workload_runs[workload_key].append(benchmark_run)

        for workload_key, runs in workload_runs.items():
            model, accelerator = workload_key
            if not runs:
                continue
            self.logger.info(f'Running additional verifiers for model: {model}, accelerator: {accelerator}')
            verifier = BenchmarkVerifier(*runs, logger=self.logger)
            category = verifier.verify()
            issues = verifier.issues
            result_dict = dict(
                multi=True,
                benchmark_run=runs,
                benchmark_type=runs[0].benchmark_type,
                benchmark_command=runs[0].command,
                benchmark_model=runs[0].model,
                issues=issues,
                category=category,
                metrics=dict()      # Add function to aggregate metrics
            )
            self.workload_results[workload_key] = Result(**result_dict)

    def print_results(self):
        print("\n========================= Results Report =========================")
        for category in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]:
            print(f"\n------------------------- {category.value.upper()} Report -------------------------")
            for result in self.run_results.values():
                if result.category == category:
                    print(f'\tRunID: {result.benchmark_run.run_id}')
                    print(f'\t    Benchmark Type: {result.benchmark_type.value}')
                    print(f'\t    Command: {result.benchmark_command}')
                    print(f'\t    Model: {result.benchmark_model}')
                    if result.issues:
                        print(f'\t    Issues:')
                        for issue in result.issues:
                            print(f'\t\t- {issue}')
                    else:
                        print(f'\t\t- No issues found')

                    if result.metrics:
                        print(f'\t    Metrics:')
                        for metric, value in result.metrics.items():
                            if type(value) in (int, float):
                                if "percentage" in metric.lower():
                                    print(f'\t\t- {metric}: {value:,.1f}%')
                                else:
                                    print(f'\t\t- {metric}: {value:,.1f}')
                            elif type(value) in (list, tuple):
                                if "percentage" in metric.lower():
                                    print(f'\t\t- {metric}: {", ".join(f"{v:,.1f}%" for v in value)}')
                                else:
                                    print(f'\t\t- {metric}: {", ".join(f"{v:,.1f}" for v in value)}')
                            else:
                                print(f'\t\t- {metric}: {value}')

                    print("\n")

        print("\n========================= Submissions Report =========================")
        for category in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]:
            print(f"\n------------------------- {category.value.upper()} Report -------------------------")
            for workload_key, workload_result in self.workload_results.items():
                if workload_result.category == category:
                    if workload_result.benchmark_model in LLM_MODELS:
                        workload_id = f"Checkpointing - {workload_result.benchmark_model}"
                    elif workload_result.benchmark_model in MODELS:
                        accelerator = workload_result.benchmark_run[0].accelerator
                        workload_id = (f"Training - {workload_result.benchmark_model}, "
                                       f"Accelerator: {accelerator}")
                    else:
                        print(f'Unknown workload type: {workload_result.benchmark_model}')

                    print(f'\tWorkloadID: {workload_id}')
                    print(f'\t    Benchmark Type: {workload_result.benchmark_type.value}')
                    if workload_result.benchmark_command:
                        print(f'\t    Command: {workload_result.benchmark_command}')

                    print(f'\t    Runs: ')
                    for run in workload_result.benchmark_run:
                        print(f'\t\t- {run.run_id} - [{self.run_results[run.run_id].category.value.upper()}]')

                    if workload_result.issues:
                        print(f'\t    Issues:')
                        for issue in workload_result.issues:
                            print(f'\t\t- {issue}')
                    else:
                        print(f'\t\t- No issues found')

                    print("\n")


    def write_json_file(self, results):
        json_file = os.path.join(self.results_dir,'results.json')
        self.logger.info(f'Writing results to {json_file}')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

    def write_csv_file(self, results):
        csv_file = os.path.join(self.results_dir,'results.csv')
        self.logger.info(f'Writing results to {csv_file}')
        flattened_results = [flatten_nested_dict(r) for r in results]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)

import csv
import json
import os.path
import pprint
import sys

from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import List, Dict, Any, Optional

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, MODELS, ACCELERATORS, CATEGORIES
from mlpstorage.rules import get_runs_files, BenchmarkVerifier, BenchmarkRun, Issue
from mlpstorage.utils import flatten_nested_dict, remove_nan_values, aggregate_run_metrics

@dataclass
class Result:
    multi: bool
    submitter: str
    system_name: str
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

        if self.args.list_checks:
            self.print_checks()
            sys.exit(EXIT_CODE.SUCCESS)

        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            self.logger.error(f'Results directory {self.results_dir} does not exist')
            sys.exit(EXIT_CODE.FILE_NOT_FOUND)

        self.run_results = dict()           # {run_id : result_dict }
        self.workload_results = dict()      # {(model) | (model, accelerator) : result_dict }
        self.submitter_overview = dict()    # {submitter: {category: {num_runs: int, num_workloads: int}}
        self.accumulate_results()

        if not self.args.no_print:
            self.print_results()

    def print_checks(self):
        from mlpstorage.rules import TrainingRunRulesChecker, CheckpointingRunRulesChecker, TrainingSubmissionRulesChecker, CheckpointSubmissionRulesChecker

        checks = dict()
        for cls in TrainingRunRulesChecker, CheckpointingRunRulesChecker, TrainingSubmissionRulesChecker, CheckpointSubmissionRulesChecker:
            checks[cls.__name__] = [item for item in dir(cls) if item.startswith('check_')]

        for checker, checks in checks.items():
            print("\nThe avilable checks for {checker}:")
            for check in checks:
                print(f'  - {check}')

    def generate_reports(self):
        # Verify the results directory exists:
        self.logger.info(f'Generating reports for {self.results_dir}')
        run_result_dicts = [report.benchmark_run.as_dict() for report in self.run_results.values()]

        if self.args.output_dir:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
            self.write_csv_file(run_result_dicts)
            self.write_json_file(run_result_dicts)
            self.write_workload_results_to_csv()
            
        return EXIT_CODE.SUCCESS

    def accumulate_results(self):
        """
        This function will look through the result_files and generate a result dictionary for each run by reading the metadata.json and summary.json files.

        If the metadata.json file does not exist, log an error and continue
        If summary.json files does not exist, set status=Failed and only use data from metadata.json the run_info from the result_files dictionary
        :return:
        """
        benchmark_runs = get_runs_files(self.results_dir, submitters=self.args.submitters,
                                        exclude=self.args.exclude_submitters, logger=self.logger)
        if self.args.models:
            benchmark_runs = [run for run in benchmark_runs if run.model in self.args.models]

        self.logger.info(f'Accumulating results from {len(benchmark_runs)} runs')
        # Process the individual runs and verify the logs against the rules
        for benchmark_run in benchmark_runs:
            self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')
            verifier = BenchmarkVerifier(benchmark_run, logger=self.logger, checks=self.args.checks)
            category = verifier.verify()
            issues = verifier.issues
            result_dict = dict(
                multi=False,
                submitter=benchmark_run.submitter,
                system_name=benchmark_run.system_name,
                benchmark_run=benchmark_run,
                benchmark_type=benchmark_run.benchmark_type,
                benchmark_command=benchmark_run.command,
                benchmark_model=benchmark_run.model,
                issues=issues,
                category=category,
                metrics=benchmark_run.metrics
            )

            if benchmark_run.submitter not in self.submitter_overview.keys():
                self.submitter_overview[benchmark_run.submitter] = dict()

            if benchmark_run.system_name not in self.submitter_overview[benchmark_run.submitter].keys():
                self.submitter_overview[benchmark_run.submitter][benchmark_run.system_name] = {cat: {"num_runs": 0, "num_workloads": 0} for cat in CATEGORIES}

            self.submitter_overview[benchmark_run.submitter][benchmark_run.system_name][category.value]['num_runs'] += 1
            self.run_results[benchmark_run.run_id] = Result(**result_dict)

        if self.args.runs_only:
            return

        # Group runs per workload to run additional verifiers
        # These will be manually defined as these checks align with a specific submission version
        # I need to group by model. For training workloads we also group by accelerator but the same checker
        # is used based on model.
        workload_runs = dict()
        systems = set()

        # "workload_runs" will contain a list of BenchmarkRun objects grouped by model and accelerator
        for benchmark_run in benchmark_runs:
            workload_key = (benchmark_run.submitter, benchmark_run.system_name, benchmark_run.model, benchmark_run.accelerator)
            if workload_key not in workload_runs.keys():
                workload_runs[workload_key] = []
            workload_runs[workload_key].append(benchmark_run)
            systems.add(benchmark_run.system_name)

        for workload_key, runs in workload_runs.items():
            submitter, system_name, model, accelerator = workload_key
            if not runs:
                continue
            self.logger.info(f'Running verifiers for Submitter: {submitter}, system: {system_name}, model: {model}, accelerator: {accelerator}')
            self.logger.debug(f'Workload runs: {pprint.pformat(runs)}')
            verifier = BenchmarkVerifier(runs, logger=self.logger, checks=self.args.checks)
            category = verifier.verify()
            issues = verifier.issues
            result_dict = dict(
                multi=True,
                submitter=runs[0].submitter,
                system_name=system_name,
                benchmark_run=runs,
                benchmark_type=runs[0].benchmark_type,
                benchmark_command=runs[0].command,
                benchmark_model=runs[0].model,
                issues=issues,
                category=category,
            )
            if len(runs) > 1:
                result_dict["metrics"] = aggregate_run_metrics([run.metrics for run in runs], agg_funcs=[mean, set])
            elif len(runs) == 1:
                result_dict["metrics"] = runs[0].metrics

            if runs[0].submitter not in self.submitter_overview.keys():
                self.submitter_overview[runs[0].submitter] = dict()

            if system_name not in self.submitter_overview[runs[0].submitter].keys():
                self.submitter_overview[runs[0].submitter][system_name] = {cat: {"num_runs": 0, "num_workloads": 0} for cat in CATEGORIES}

            self.submitter_overview[runs[0].submitter][system_name][category.value]['num_workloads'] += 1
            self.workload_results[workload_key] = Result(**result_dict)

        self.logger.result(f'Found {len(workload_runs)} workloads across {len(systems)} systems')

    def print_results(self):
        print("\n========================= Results Report =========================")
        print("This report represents individual runs of benchmarks")
        for category in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]:
            print(f"\n------------------------- {category.value.upper()} Report -------------------------")
            for result in self.run_results.values():
                if result.category == category:
                    print(f'\tRunID: {result.benchmark_run.run_id}')
                    print(f'\t    Submitter: {result.benchmark_run.submitter}')
                    print(f'\t    System Name: {result.benchmark_run.system_name}')
                    print(f'\t    Benchmark Type: {result.benchmark_type.value}')
                    print(f'\t    Command: {result.benchmark_command}')
                    print(f'\t    Model: {result.benchmark_model}')

                    if self.args.metadata_only:
                        print("\n")
                        continue

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

        if not self.args.runs_only:
            print("\n========================= Submissions Report =========================")
            print("This report represents aggregated runs of benchmarks for submissions")
            submitters = []
            for workload_key in self.workload_results.keys():
                submitter, system_name, model, accelerator = workload_key
                if submitter not in submitters:
                    submitters.append(submitter)

            for category in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]:
                print(f"\n------------------------- {category.value.upper()} Report -------------------------")
                for submitter in submitters:
                    for workload_key, workload_result in self.workload_results.items():
                        if workload_result.category == category and workload_key[0] == submitter:
                            submitter, system_name, model, accelerator = workload_key
                            if workload_result.benchmark_model in LLM_MODELS:
                                workload_id = f"Checkpointing - {workload_result.benchmark_model}"
                            elif workload_result.benchmark_model in MODELS:
                                accelerator = workload_result.benchmark_run[0].accelerator
                                workload_id = (f"Training - {workload_result.benchmark_model}, "
                                               f"Accelerator: {accelerator}")
                            else:
                                print(f'Unknown workload type: {workload_result.benchmark_model}')

                            print(f'\tWorkloadID: {workload_id}')
                            print(f'\t    Submitter: {submitter}')
                            print(f'\t    System Name: {system_name}')
                            print(f'\t    Benchmark Type: {workload_result.benchmark_type.value}')
                            if workload_result.benchmark_command:
                                print(f'\t    Command: {workload_result.benchmark_command}')

                            if self.args.metadata_only:
                                print(f'\t    Run0 ID: {workload_result.benchmark_run[0].run_id}')
                                print("\n")
                                continue

                            print(f'\t    Runs: ')
                            for run in workload_result.benchmark_run:
                                print(f'\t\t- {run.run_id} - [{self.run_results[run.run_id].category.value.upper()}]')

                            if workload_result.issues:
                                print(f'\t    Issues:')
                                for issue in workload_result.issues:
                                    print(f'\t\t- {issue}')
                            else:
                                print(f'\t\t- No workload issues found')

                            for benchmark_run in workload_result.benchmark_run:
                                for issue in benchmark_run.issues:
                                    if issue.validation == PARAM_VALIDATION.INVALID:
                                        print(f'\t\t- {issue} ({benchmark_run.run_id})')

                            if workload_result.metrics:
                                print(f'\t    Metrics:')
                                for metric, value in sorted(workload_result.metrics.items()):
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

        print("\n========================= Submitter Overview Report =========================")
        grand_total_workloads = {cat: 0 for cat in CATEGORIES}
        grand_total_runs = {cat: 0 for cat in CATEGORIES}
        for submitter, overview in self.submitter_overview.items():
            print(f"\n------------------------- {submitter} Report -------------------------")
            total_workloads = {cat: 0 for cat in CATEGORIES}
            total_runs = {cat: 0 for cat in CATEGORIES}
            for system_name, category_count in overview.items():
                print(f'\tSystem Name: {system_name}')
                for category in CATEGORIES:
                    counts = category_count.get(category, {"num_workloads": 0, "num_runs": 0})
                    total_workloads[category] += counts["num_workloads"]
                    total_runs[category] += counts["num_runs"]
                    grand_total_workloads[category] += counts["num_workloads"]
                    grand_total_runs[category] += counts["num_runs"]
                    print(f'\t    {category.upper()} Workloads:\t{counts["num_workloads"]}    (Runs: {counts["num_runs"]})')
                print()

            for cat in CATEGORIES:
                print(f'\nTotal {cat.upper()} Workloads: {total_workloads[cat]}    ')
                print(f'Total {cat.upper()} Runs: {total_runs[cat]}')

        for cat in CATEGORIES:
            print(f'\nGrand Total {cat.upper()} Workloads: {grand_total_workloads[cat]}    ')
            print(f'Grand Total {cat.upper()} Runs: {grand_total_runs[cat]}')

    def write_json_file(self, results):
        json_file = os.path.join(self.args.output_dir,'results.json')
        self.logger.info(f'Writing results to {json_file}')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

    def write_csv_file(self, results):
        csv_file = os.path.join(self.args.output_dir,'results.csv')
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

    def write_workload_results_to_csv(self):
        csv_file = os.path.join(self.args.output_dir, 'workload_results.csv')
        result_dicts = []

        # This line will pause the execution and allow you to inspect the variables in the current scope.
        for key, result in self.workload_results.items():
            result_dict = result.__dict__
            result_dict['benchmark_type'] = result.benchmark_type.name
            result_dict['Division'] = result.benchmark_run[0].submitted_category
            result_dict['number_of_nodes'] = result.benchmark_run[0].num_hosts
            result_dict['# Simulated Accelerators'] = result.benchmark_run[0].num_processes
            result_dict['Submitter'] = key[0]
            result_dict['System Name'] = key[1]
            result_dict['Model'] = key[2]
            result_dict['Accelerator'] = key[3]
            result_dict['run_ids'] = [run.run_id for run in result.benchmark_run]
            result_dict['num_runs'] = len(result.benchmark_run)

            if result.benchmark_type == BENCHMARK_TYPES.checkpointing.name:
                if result.metrics.get('calculated_mean_load_throughput_from_strict_times'):
                    result_dict['Read B/W (GiB/s)'] = result.metrics.get('calculated_mean_load_throughput_from_strict_times', None)
                    result_dict['Write B/W (GiB/s)'] = result.metrics.get('calculated_mean_save_throughput_from_strict_times', None)
                else:
                    result_dict['Read B/W (GiB/s)'] = result.metrics.get('mean_calculated_mean_load_throughput_from_strict_times', None)
                    result_dict['Write B/W (GiB/s)'] = result.metrics.get('mean_calculated_mean_save_throughput_from_strict_times', None)
                if result.metrics.get('mean_of_max_save_duration'):
                    result_dict['Read Duration'] = result.metrics.get('mean_of_max_load_duration', None)
                    result_dict['Write Duration'] = result.metrics.get('mean_of_max_save_duration', None)
                else:
                    result_dict['Read Duration'] = result.metrics.get('mean_mean_of_max_load_duration', None)
                    result_dict['Write Duration'] = result.metrics.get('mean_mean_of_max_save_duration', None)

            elif result.benchmark_type == BENCHMARK_TYPES.training.name:
                result_dict['Read B/W (GiB/s)'] = result.metrics.get('mean_train_io_mean_MB_per_second', None)


            result_dict['System'] = dict()
            # import pdb
            # pdb.set_trace()
            if isinstance(result.benchmark_run[0].system_description['System'], dict):
                for k, v in result.benchmark_run[0].system_description['System'].items():
                    if isinstance(v, str) and (":" in v or v.count('.') > 1):
                        continue
                    result_dict['System'][k] = v


            result_dicts.append(result_dict)

        flattened_results = [flatten_nested_dict(r) for r in result_dicts]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)
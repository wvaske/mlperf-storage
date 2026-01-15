"""
CSV format handler for MLPerf Storage benchmark results.

Provides flat CSV file export for data analysis.
"""

import csv
import io
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from mlpstorage.reporting.formats import ReportFormat, FormatRegistry, FormatConfig

if TYPE_CHECKING:
    from mlpstorage.reporting import Result


@FormatRegistry.register
class CSVFormat(ReportFormat):
    """Format results as CSV files for data analysis."""

    def __init__(self, config: Optional[FormatConfig] = None):
        """
        Initialize the CSV formatter.

        Args:
            config: Optional configuration.
        """
        super().__init__(config)

    @property
    def name(self) -> str:
        return "csv"

    @property
    def extension(self) -> str:
        return "csv"

    @property
    def content_type(self) -> str:
        return "text/csv"

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _remove_nan_values(self, d: Dict) -> Dict:
        """Remove NaN and None values from dictionary."""
        result = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, float):
                import math
                if math.isnan(v) or math.isinf(v):
                    continue
            result[k] = v
        return result

    def _result_to_row(self, result: 'Result') -> Dict[str, Any]:
        """Convert a Result object to a flat dictionary row."""
        run = result.benchmark_run

        row = {
            'run_id': str(run.run_id) if hasattr(run, 'run_id') else '',
            'benchmark_type': result.benchmark_type.value if result.benchmark_type else '',
            'model': str(result.benchmark_model) if result.benchmark_model else '',
            'command': result.benchmark_command or '',
            'category': result.category.value if result.category else '',
            'num_processes': run.num_processes if hasattr(run, 'num_processes') else '',
            'accelerator': str(run.accelerator) if hasattr(run, 'accelerator') and run.accelerator else '',
            'run_datetime': run.run_datetime if hasattr(run, 'run_datetime') else '',
        }

        # Add flattened metrics
        if result.metrics:
            metrics_flat = self._flatten_dict(result.metrics, 'metrics')
            row.update(metrics_flat)

        # Add flattened parameters (if advanced mode)
        if self.config.include_advanced and hasattr(run, 'parameters') and run.parameters:
            params_flat = self._flatten_dict(run.parameters, 'params')
            row.update(params_flat)

        # Add cluster info (if requested)
        if self.config.include_cluster_info and hasattr(run, 'system_info') and run.system_info:
            system_info = run.system_info
            if hasattr(system_info, 'as_dict'):
                info_dict = system_info.as_dict()
                row['cluster.num_hosts'] = info_dict.get('num_hosts', '')
                row['cluster.total_memory_gb'] = info_dict.get('total_memory_bytes', 0) / (1024**3)
                row['cluster.total_cores'] = info_dict.get('total_cores', '')

        # Add issue summary
        row['num_issues'] = len(result.issues)
        row['num_invalid_issues'] = len([i for i in result.issues
                                         if hasattr(i, 'validation') and i.validation.value == 'invalid'])

        return self._remove_nan_values(row)

    def generate_runs_csv(self, results: List['Result']) -> str:
        """
        Generate CSV content for benchmark runs.

        Args:
            results: List of Result objects.

        Returns:
            CSV content as string.
        """
        if not results:
            return ""

        rows = [self._result_to_row(r) for r in results]

        # Collect all field names
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())

        # Sort fieldnames for consistent output
        sorted_fields = sorted(fieldnames)

        # Prioritize certain fields at the beginning
        priority_fields = ['run_id', 'benchmark_type', 'model', 'command', 'category',
                          'num_processes', 'accelerator', 'run_datetime']
        ordered_fields = [f for f in priority_fields if f in sorted_fields]
        ordered_fields.extend(f for f in sorted_fields if f not in priority_fields)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=ordered_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

    def generate_workloads_csv(self, workload_results: Dict) -> str:
        """
        Generate CSV content for workload submissions.

        Args:
            workload_results: Dictionary of workload results.

        Returns:
            CSV content as string.
        """
        if not workload_results:
            return ""

        rows = []
        for (model, accelerator), result in workload_results.items():
            num_runs = len(result.benchmark_run) if isinstance(result.benchmark_run, list) else 1

            row = {
                'model': str(model) if model else '',
                'accelerator': str(accelerator) if accelerator else '',
                'benchmark_type': result.benchmark_type.value if result.benchmark_type else '',
                'command': result.benchmark_command or '',
                'category': result.category.value if result.category else '',
                'num_runs': num_runs,
                'num_issues': len(result.issues),
            }
            rows.append(row)

        fieldnames = ['model', 'accelerator', 'benchmark_type', 'command', 'category',
                     'num_runs', 'num_issues']

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

    def generate_metrics_csv(self, results: List['Result']) -> str:
        """
        Generate CSV content for performance metrics.

        Args:
            results: List of Result objects.

        Returns:
            CSV content as string.
        """
        if not results:
            return ""

        # Collect all metric keys
        all_keys = set()
        for result in results:
            if result.metrics:
                flat_metrics = self._flatten_dict(result.metrics)
                all_keys.update(flat_metrics.keys())

        if not all_keys:
            return ""

        rows = []
        for result in results:
            run = result.benchmark_run
            row = {
                'run_id': str(run.run_id) if hasattr(run, 'run_id') else '',
                'model': str(result.benchmark_model) if result.benchmark_model else '',
                'category': result.category.value if result.category else '',
            }

            if result.metrics:
                flat_metrics = self._flatten_dict(result.metrics)
                row.update(flat_metrics)

            rows.append(self._remove_nan_values(row))

        # Sort field names
        fieldnames = ['run_id', 'model', 'category'] + sorted(all_keys)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n',
                               extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

    def generate_param_ranges_csv(self, param_ranges: Dict[str, Dict]) -> str:
        """
        Generate CSV content for parameter ranges.

        Args:
            param_ranges: Dictionary of parameter ranges by workload.

        Returns:
            CSV content as string.
        """
        if not param_ranges:
            return ""

        rows = []
        for workload, ranges in param_ranges.items():
            for param, stats in ranges.items():
                if isinstance(stats, dict):
                    row = {
                        'workload': workload,
                        'parameter': param,
                        'min': stats.get('min', ''),
                        'max': stats.get('max', ''),
                        'avg': stats.get('avg', ''),
                        'count': len(stats.get('values', [])),
                    }
                    rows.append(row)

        fieldnames = ['workload', 'parameter', 'min', 'max', 'avg', 'count']

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

    def generate(self, results: List['Result'], workload_results: Dict,
                 advanced_data: Optional[Dict] = None) -> bytes:
        """
        Generate CSV report (runs CSV as primary output).

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.
            advanced_data: Optional advanced data.

        Returns:
            CSV content as bytes.
        """
        return self.generate_runs_csv(results).encode('utf-8')

    def generate_all_csvs(self, results: List['Result'], workload_results: Dict,
                          output_dir: str, advanced_data: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate all CSV files.

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.
            output_dir: Directory to write files.
            advanced_data: Optional advanced data.

        Returns:
            Dictionary mapping file type to file path.
        """
        import os

        files = {}

        # Runs CSV
        runs_path = os.path.join(output_dir, 'results_runs.csv')
        with open(runs_path, 'w') as f:
            f.write(self.generate_runs_csv(results))
        files['runs'] = runs_path

        # Workloads CSV
        if workload_results:
            workloads_path = os.path.join(output_dir, 'results_workloads.csv')
            with open(workloads_path, 'w') as f:
                f.write(self.generate_workloads_csv(workload_results))
            files['workloads'] = workloads_path

        # Metrics CSV
        metrics_csv = self.generate_metrics_csv(results)
        if metrics_csv:
            metrics_path = os.path.join(output_dir, 'results_metrics.csv')
            with open(metrics_path, 'w') as f:
                f.write(metrics_csv)
            files['metrics'] = metrics_path

        # Parameter ranges CSV (if advanced)
        if advanced_data and 'param_ranges' in advanced_data:
            ranges_path = os.path.join(output_dir, 'results_param_ranges.csv')
            with open(ranges_path, 'w') as f:
                f.write(self.generate_param_ranges_csv(advanced_data['param_ranges']))
            files['param_ranges'] = ranges_path

        return files

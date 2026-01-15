"""
JSON format handler for MLPerf Storage benchmark results.

Provides JSON export for programmatic access.
"""

import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from mlpstorage.reporting.formats import ReportFormat, FormatRegistry, FormatConfig
from mlpstorage.config import PARAM_VALIDATION

if TYPE_CHECKING:
    from mlpstorage.reporting import Result


class MLPSJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MLPerf Storage objects."""

    def default(self, obj):
        # Handle enums
        if hasattr(obj, 'value'):
            return obj.value

        # Handle objects with as_dict method
        if hasattr(obj, 'as_dict'):
            return obj.as_dict()

        # Handle objects with info property
        if hasattr(obj, 'info'):
            return obj.info

        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}

        return super().default(obj)


@FormatRegistry.register
class JSONFormat(ReportFormat):
    """Format results as JSON for programmatic access."""

    def __init__(self, config: Optional[FormatConfig] = None, indent: int = 2):
        """
        Initialize the JSON formatter.

        Args:
            config: Optional configuration.
            indent: JSON indentation level.
        """
        super().__init__(config)
        self.indent = indent

    @property
    def name(self) -> str:
        return "json"

    @property
    def extension(self) -> str:
        return "json"

    @property
    def content_type(self) -> str:
        return "application/json"

    def _result_to_dict(self, result: 'Result') -> Dict[str, Any]:
        """Convert a Result object to a dictionary."""
        run = result.benchmark_run

        data = {
            'run_id': str(run.run_id) if hasattr(run, 'run_id') else None,
            'benchmark_type': result.benchmark_type.value if result.benchmark_type else None,
            'model': str(result.benchmark_model) if result.benchmark_model else None,
            'command': result.benchmark_command,
            'category': result.category.value if result.category else None,
            'metrics': result.metrics,
            'issues': [
                {
                    'validation': i.validation.value if hasattr(i, 'validation') else None,
                    'message': i.message if hasattr(i, 'message') else str(i),
                    'parameter': getattr(i, 'parameter', None),
                    'expected': getattr(i, 'expected', None),
                    'actual': getattr(i, 'actual', None),
                }
                for i in result.issues
            ],
        }

        # Add run details
        if hasattr(run, 'num_processes'):
            data['num_processes'] = run.num_processes
        if hasattr(run, 'accelerator') and run.accelerator:
            data['accelerator'] = str(run.accelerator)
        if hasattr(run, 'run_datetime'):
            data['run_datetime'] = run.run_datetime

        # Add parameters if advanced mode
        if self.config.include_advanced:
            if hasattr(run, 'parameters'):
                data['parameters'] = run.parameters
            if hasattr(run, 'override_parameters'):
                data['override_parameters'] = run.override_parameters

        # Add cluster info if requested
        if self.config.include_cluster_info and hasattr(run, 'system_info') and run.system_info:
            if hasattr(run.system_info, 'as_dict'):
                data['system_info'] = run.system_info.as_dict()

        return data

    def _workload_to_dict(self, workload_key: tuple, result: 'Result') -> Dict[str, Any]:
        """Convert a workload result to a dictionary."""
        model, accelerator = workload_key

        runs = result.benchmark_run if isinstance(result.benchmark_run, list) else [result.benchmark_run]

        return {
            'model': str(model) if model else None,
            'accelerator': str(accelerator) if accelerator else None,
            'benchmark_type': result.benchmark_type.value if result.benchmark_type else None,
            'command': result.benchmark_command,
            'category': result.category.value if result.category else None,
            'num_runs': len(runs),
            'run_ids': [str(r.run_id) if hasattr(r, 'run_id') else str(r) for r in runs],
            'issues': [
                {
                    'validation': i.validation.value if hasattr(i, 'validation') else None,
                    'message': i.message if hasattr(i, 'message') else str(i),
                    'parameter': getattr(i, 'parameter', None),
                }
                for i in result.issues
            ],
        }

    def generate(self, results: List['Result'], workload_results: Dict,
                 advanced_data: Optional[Dict] = None) -> bytes:
        """
        Generate JSON report.

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.
            advanced_data: Optional advanced data.

        Returns:
            JSON content as bytes.
        """
        # Calculate summary
        closed = sum(1 for r in results if r.category == PARAM_VALIDATION.CLOSED)
        open_count = sum(1 for r in results if r.category == PARAM_VALIDATION.OPEN)
        invalid = sum(1 for r in results if r.category == PARAM_VALIDATION.INVALID)

        report = {
            'summary': {
                'total_runs': len(results),
                'closed_runs': closed,
                'open_runs': open_count,
                'invalid_runs': invalid,
                'total_workloads': len(workload_results),
            },
            'runs': [self._result_to_dict(r) for r in results],
            'workloads': [
                self._workload_to_dict(k, v)
                for k, v in workload_results.items()
            ],
        }

        # Add advanced data if provided
        if advanced_data:
            if 'param_ranges' in advanced_data:
                report['parameter_ranges'] = advanced_data['param_ranges']
            if 'cluster_info' in advanced_data:
                report['cluster_info'] = advanced_data['cluster_info']

        return json.dumps(report, indent=self.indent, cls=MLPSJSONEncoder).encode('utf-8')

    def generate_pretty(self, results: List['Result'], workload_results: Dict,
                        advanced_data: Optional[Dict] = None) -> str:
        """
        Generate pretty-printed JSON string.

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.
            advanced_data: Optional advanced data.

        Returns:
            Pretty-printed JSON string.
        """
        return self.generate(results, workload_results, advanced_data).decode('utf-8')

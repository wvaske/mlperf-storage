"""
Advanced data collector for MLPerf Storage benchmark reports.

Collects extended data including parameter ranges and cluster information
for advanced/debug output mode.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from mlpstorage.rules.models import BenchmarkRun


class AdvancedDataCollector:
    """Collect extended data for advanced output mode."""

    def __init__(self, logger=None):
        """
        Initialize the collector.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger

    def collect_all(self, benchmark_runs: List['BenchmarkRun'],
                   include_param_ranges: bool = True,
                   include_cluster_info: bool = True) -> Dict[str, Any]:
        """
        Collect all advanced data.

        Args:
            benchmark_runs: List of BenchmarkRun objects.
            include_param_ranges: Whether to include parameter ranges.
            include_cluster_info: Whether to include cluster information.

        Returns:
            Dictionary of collected advanced data.
        """
        data = {}

        if include_param_ranges:
            data['param_ranges'] = self.collect_param_ranges(benchmark_runs)

        if include_cluster_info:
            data['cluster_info'] = self.collect_cluster_details(benchmark_runs)

        return data

    def collect_param_ranges(self, benchmark_runs: List['BenchmarkRun']) -> Dict[str, Dict]:
        """
        Calculate min/max/avg for each numeric parameter.

        Groups parameters by workload (model + benchmark_type).

        Args:
            benchmark_runs: List of BenchmarkRun objects.

        Returns:
            Dictionary of parameter ranges by workload.
        """
        workload_params = defaultdict(lambda: defaultdict(list))

        for run in benchmark_runs:
            # Create workload key
            benchmark_type = run.benchmark_type.value if hasattr(run, 'benchmark_type') and run.benchmark_type else 'unknown'
            model = str(run.model) if hasattr(run, 'model') and run.model else 'unknown'
            workload_key = f"{benchmark_type}_{model}"

            # Collect parameters
            params = {}
            if hasattr(run, 'parameters') and run.parameters:
                params = self._flatten_dict(run.parameters)

            # Also collect metrics
            if hasattr(run, 'metrics') and run.metrics:
                metrics = self._flatten_dict(run.metrics, prefix='metrics')
                params.update(metrics)

            for param, value in params.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    workload_params[workload_key][param].append(value)

        # Calculate statistics
        result = {}
        for workload, params in workload_params.items():
            result[workload] = {}
            for param, values in params.items():
                if values:
                    result[workload][param] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'values': values,
                    }

        return result

    def collect_cluster_details(self, benchmark_runs: List['BenchmarkRun']) -> Optional[Dict]:
        """
        Extract detailed cluster configuration from runs.

        Args:
            benchmark_runs: List of BenchmarkRun objects.

        Returns:
            Dictionary of cluster information or None if not available.
        """
        # Find a run with cluster information
        for run in benchmark_runs:
            if hasattr(run, 'system_info') and run.system_info:
                system_info = run.system_info
                if hasattr(system_info, 'as_dict'):
                    return system_info.as_dict()
                elif hasattr(system_info, 'to_detailed_dict'):
                    return system_info.to_detailed_dict()

        # Try to aggregate from multiple runs
        hosts_seen = {}
        total_memory = 0
        total_cores = 0

        for run in benchmark_runs:
            if hasattr(run, 'system_info') and run.system_info:
                info = run.system_info
                if hasattr(info, 'host_info_list'):
                    for host in info.host_info_list:
                        hostname = host.hostname if hasattr(host, 'hostname') else str(host)
                        if hostname not in hosts_seen:
                            hosts_seen[hostname] = {
                                'hostname': hostname,
                                'memory': host.memory.total if hasattr(host, 'memory') and host.memory else 0,
                                'cpu': {
                                    'model': host.cpu.model if hasattr(host, 'cpu') and host.cpu else '',
                                    'num_cores': host.cpu.num_cores if hasattr(host, 'cpu') and host.cpu else 0,
                                } if hasattr(host, 'cpu') and host.cpu else {},
                            }
                            if hasattr(host, 'memory') and host.memory:
                                total_memory += host.memory.total
                            if hasattr(host, 'cpu') and host.cpu:
                                total_cores += host.cpu.num_cores

        if hosts_seen:
            return {
                'num_hosts': len(hosts_seen),
                'total_memory_bytes': total_memory,
                'total_cores': total_cores,
                'hosts': list(hosts_seen.values()),
            }

        return None

    def collect_timing_analysis(self, benchmark_runs: List['BenchmarkRun']) -> Dict[str, Any]:
        """
        Analyze timing between runs and epochs.

        Args:
            benchmark_runs: List of BenchmarkRun objects.

        Returns:
            Dictionary of timing analysis.
        """
        timing_data = {
            'run_times': [],
            'epoch_times': [],
            'inter_run_gaps': [],
        }

        sorted_runs = sorted(
            benchmark_runs,
            key=lambda r: r.run_datetime if hasattr(r, 'run_datetime') else ''
        )

        for i, run in enumerate(sorted_runs):
            # Collect run timing
            if hasattr(run, 'metrics') and run.metrics:
                metrics = run.metrics
                if 'duration' in metrics or 'total_time' in metrics:
                    duration = metrics.get('duration', metrics.get('total_time'))
                    timing_data['run_times'].append({
                        'run_id': str(run.run_id) if hasattr(run, 'run_id') else '',
                        'duration': duration,
                    })

                # Collect epoch timing
                if 'epoch_times' in metrics:
                    timing_data['epoch_times'].extend(metrics['epoch_times'])

        return timing_data

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.', prefix: str = '') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        if prefix:
            parent_key = f"{prefix}{sep}" if not parent_key else f"{prefix}{sep}{parent_key}"

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                # Skip lists for parameter ranges
                pass
            else:
                items.append((new_key, v))

        return dict(items)


def collect_advanced_data(results: List, benchmark_runs: List['BenchmarkRun'] = None,
                         include_param_ranges: bool = True,
                         include_cluster_info: bool = True,
                         logger=None) -> Dict[str, Any]:
    """
    Convenience function to collect advanced data.

    Args:
        results: List of Result objects.
        benchmark_runs: Optional list of BenchmarkRun objects.
        include_param_ranges: Whether to include parameter ranges.
        include_cluster_info: Whether to include cluster information.
        logger: Optional logger.

    Returns:
        Dictionary of advanced data.
    """
    # Extract benchmark runs from results if not provided
    if benchmark_runs is None:
        benchmark_runs = []
        for result in results:
            if hasattr(result, 'benchmark_run'):
                run = result.benchmark_run
                if isinstance(run, list):
                    benchmark_runs.extend(run)
                else:
                    benchmark_runs.append(run)

    collector = AdvancedDataCollector(logger=logger)
    return collector.collect_all(
        benchmark_runs,
        include_param_ranges=include_param_ranges,
        include_cluster_info=include_cluster_info
    )

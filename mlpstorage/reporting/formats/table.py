"""
Table format handler for MLPerf Storage benchmark results.

Provides formatted table output for terminal display.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from mlpstorage.config import PARAM_VALIDATION

if TYPE_CHECKING:
    from mlpstorage.reporting import Result


class TableFormat:
    """Format results as tables for terminal display."""

    # Table style configurations
    STYLES = {
        'simple': {
            'horizontal': '-',
            'vertical': '|',
            'corner': '+',
            'header_sep': '-',
        },
        'grid': {
            'horizontal': '-',
            'vertical': '|',
            'corner': '+',
            'header_sep': '=',
        },
        'minimal': {
            'horizontal': '',
            'vertical': '  ',
            'corner': '',
            'header_sep': '-',
        },
    }

    # Terminal colors
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'cyan': '\033[96m',
    }

    def __init__(self, style: str = 'simple', use_colors: bool = True,
                 max_width: int = 120):
        """
        Initialize the table formatter.

        Args:
            style: Table style ('simple', 'grid', 'minimal').
            use_colors: Whether to use terminal colors.
            max_width: Maximum table width in characters.
        """
        self.style = self.STYLES.get(style, self.STYLES['simple'])
        self.use_colors = use_colors
        self.max_width = max_width

    @property
    def name(self) -> str:
        return "table"

    @property
    def extension(self) -> str:
        return "txt"

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def _category_color(self, category: PARAM_VALIDATION) -> str:
        """Get color for a category."""
        if category == PARAM_VALIDATION.CLOSED:
            return 'green'
        elif category == PARAM_VALIDATION.OPEN:
            return 'yellow'
        return 'red'

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + '...'

    def _calculate_column_widths(self, headers: List[str], rows: List[List[str]],
                                 min_width: int = 5) -> List[int]:
        """Calculate optimal column widths."""
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    # Strip ANSI codes for width calculation
                    clean_cell = self._strip_ansi(str(cell))
                    widths[i] = max(widths[i], len(clean_cell))

        # Apply minimum width
        widths = [max(w, min_width) for w in widths]

        # Adjust if total exceeds max_width
        total = sum(widths) + len(widths) * 3  # Account for separators
        if total > self.max_width and len(widths) > 0:
            excess = total - self.max_width
            # Reduce widest columns first
            while excess > 0:
                max_idx = widths.index(max(widths))
                if widths[max_idx] > min_width:
                    widths[max_idx] -= 1
                    excess -= 1
                else:
                    break

        return widths

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI color codes from text."""
        import re
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_pattern.sub('', text)

    def _format_row(self, cells: List[str], widths: List[int]) -> str:
        """Format a single row."""
        sep = self.style['vertical']
        formatted = []
        for cell, width in zip(cells, widths):
            clean_cell = self._strip_ansi(str(cell))
            padding = width - len(clean_cell)
            formatted.append(f" {cell}{' ' * padding} ")
        return sep + sep.join(formatted) + sep

    def _format_separator(self, widths: List[int], char: str = None) -> str:
        """Format a horizontal separator."""
        if char is None:
            char = self.style['horizontal']
        if not char:
            return ""
        corner = self.style['corner']
        segments = [char * (w + 2) for w in widths]
        return corner + corner.join(segments) + corner

    def format_runs_table(self, results: List['Result']) -> str:
        """
        Generate formatted table of benchmark runs.

        Args:
            results: List of Result objects.

        Returns:
            Formatted table string.
        """
        if not results:
            return "No runs to display."

        headers = ['Run ID', 'Type', 'Model', 'Command', 'Category', 'AU %', 'Throughput']

        rows = []
        for result in results:
            run = result.benchmark_run
            category = result.category
            category_str = self._color(
                category.value.upper(),
                self._category_color(category)
            )

            # Extract metrics
            metrics = result.metrics or {}
            au_pct = metrics.get('au', metrics.get('accelerator_utilization', '-'))
            if isinstance(au_pct, (int, float)):
                au_pct = f"{au_pct:.1f}%"

            throughput = metrics.get('throughput', '-')
            if isinstance(throughput, (int, float)):
                throughput = f"{throughput:.2f}"

            rows.append([
                str(run.run_id) if hasattr(run, 'run_id') else str(run),
                result.benchmark_type.value if result.benchmark_type else '-',
                str(result.benchmark_model) if result.benchmark_model else '-',
                result.benchmark_command or '-',
                category_str,
                str(au_pct),
                str(throughput),
            ])

        widths = self._calculate_column_widths(headers, rows)

        lines = []
        lines.append(self._format_separator(widths))
        lines.append(self._format_row(headers, widths))
        lines.append(self._format_separator(widths, self.style['header_sep']))

        for row in rows:
            lines.append(self._format_row(row, widths))

        lines.append(self._format_separator(widths))

        return '\n'.join(lines)

    def format_workloads_table(self, workload_results: Dict) -> str:
        """
        Generate formatted table of workload submissions.

        Args:
            workload_results: Dictionary of workload results.

        Returns:
            Formatted table string.
        """
        if not workload_results:
            return "No workload submissions to display."

        headers = ['Model', 'Accelerator', 'Runs', 'Category', 'Issues']

        rows = []
        for (model, accelerator), result in workload_results.items():
            category = result.category
            category_str = self._color(
                category.value.upper(),
                self._category_color(category)
            )

            num_runs = len(result.benchmark_run) if isinstance(result.benchmark_run, list) else 1
            num_issues = len([i for i in result.issues
                            if i.validation != PARAM_VALIDATION.CLOSED])

            rows.append([
                str(model) if model else '-',
                str(accelerator) if accelerator else '-',
                str(num_runs),
                category_str,
                str(num_issues),
            ])

        widths = self._calculate_column_widths(headers, rows)

        lines = []
        lines.append(self._format_separator(widths))
        lines.append(self._format_row(headers, widths))
        lines.append(self._format_separator(widths, self.style['header_sep']))

        for row in rows:
            lines.append(self._format_row(row, widths))

        lines.append(self._format_separator(widths))

        return '\n'.join(lines)

    def format_metrics_table(self, results: List['Result']) -> str:
        """
        Generate formatted table of performance metrics.

        Args:
            results: List of Result objects.

        Returns:
            Formatted table string.
        """
        if not results:
            return "No metrics to display."

        # Collect all metric keys
        all_keys = set()
        for result in results:
            if result.metrics:
                all_keys.update(result.metrics.keys())

        if not all_keys:
            return "No metrics available."

        # Filter to common metrics
        priority_keys = ['au', 'throughput', 'train_au_percentage', 'train_throughput_samples_per_second',
                        'samples_processed', 'epochs_completed']
        headers = ['Run ID'] + [k for k in priority_keys if k in all_keys]

        rows = []
        for result in results:
            run_id = str(result.benchmark_run.run_id) if hasattr(result.benchmark_run, 'run_id') else '-'
            row = [self._truncate(run_id, 30)]

            metrics = result.metrics or {}
            for key in headers[1:]:
                value = metrics.get(key, '-')
                if isinstance(value, float):
                    value = f"{value:.2f}"
                row.append(str(value))

            rows.append(row)

        widths = self._calculate_column_widths(headers, rows)

        lines = []
        lines.append(self._format_separator(widths))
        lines.append(self._format_row(headers, widths))
        lines.append(self._format_separator(widths, self.style['header_sep']))

        for row in rows:
            lines.append(self._format_row(row, widths))

        lines.append(self._format_separator(widths))

        return '\n'.join(lines)

    def format_parameter_ranges_table(self, param_ranges: Dict[str, Dict]) -> str:
        """
        Generate formatted table of parameter ranges.

        Args:
            param_ranges: Dictionary of parameter ranges by workload.

        Returns:
            Formatted table string.
        """
        if not param_ranges:
            return "No parameter ranges to display."

        headers = ['Parameter', 'Min', 'Max', 'Avg', 'Values']

        all_rows = []
        for workload, ranges in param_ranges.items():
            all_rows.append([self._color(f"=== {workload} ===", 'bold'), '', '', '', ''])

            for param, stats in ranges.items():
                if isinstance(stats, dict):
                    min_val = stats.get('min', '-')
                    max_val = stats.get('max', '-')
                    avg_val = stats.get('avg', '-')
                    values = stats.get('values', [])

                    if isinstance(min_val, float):
                        min_val = f"{min_val:.2f}"
                    if isinstance(max_val, float):
                        max_val = f"{max_val:.2f}"
                    if isinstance(avg_val, float):
                        avg_val = f"{avg_val:.2f}"

                    values_str = ', '.join(str(v) for v in values[:5])
                    if len(values) > 5:
                        values_str += '...'

                    all_rows.append([param, str(min_val), str(max_val), str(avg_val), values_str])

        widths = self._calculate_column_widths(headers, all_rows)

        lines = []
        lines.append(self._format_separator(widths))
        lines.append(self._format_row(headers, widths))
        lines.append(self._format_separator(widths, self.style['header_sep']))

        for row in all_rows:
            lines.append(self._format_row(row, widths))

        lines.append(self._format_separator(widths))

        return '\n'.join(lines)

    def format_cluster_info_table(self, cluster_info: Dict) -> str:
        """
        Generate formatted table of cluster information.

        Args:
            cluster_info: Dictionary of cluster information.

        Returns:
            Formatted table string.
        """
        if not cluster_info:
            return "No cluster information to display."

        headers = ['Hostname', 'CPU Model', 'Cores', 'Memory (GB)', 'Status']

        rows = []
        hosts = cluster_info.get('hosts', [])
        for host in hosts:
            hostname = host.get('hostname', '-')
            cpu = host.get('cpu', {})
            cpu_model = cpu.get('model', '-')[:30] if cpu else '-'
            cores = cpu.get('num_cores', '-') if cpu else '-'

            memory = host.get('memory', {})
            mem_gb = memory.get('total', 0) / (1024**3) if memory else 0
            mem_str = f"{mem_gb:.1f}" if mem_gb > 0 else '-'

            status = self._color('OK', 'green')

            rows.append([hostname, cpu_model, str(cores), mem_str, status])

        # Add summary row
        total_mem = cluster_info.get('total_memory_bytes', 0) / (1024**3)
        total_cores = cluster_info.get('total_cores', 0)
        num_hosts = cluster_info.get('num_hosts', len(hosts))

        rows.append([
            self._color('TOTAL', 'bold'),
            f"{num_hosts} hosts",
            str(total_cores),
            f"{total_mem:.1f}",
            ''
        ])

        widths = self._calculate_column_widths(headers, rows)

        lines = []
        lines.append(self._format_separator(widths))
        lines.append(self._format_row(headers, widths))
        lines.append(self._format_separator(widths, self.style['header_sep']))

        for row in rows[:-1]:
            lines.append(self._format_row(row, widths))

        lines.append(self._format_separator(widths))
        lines.append(self._format_row(rows[-1], widths))
        lines.append(self._format_separator(widths))

        return '\n'.join(lines)

    def format_summary_table(self, results: List['Result'], workload_results: Dict) -> str:
        """
        Generate a summary table.

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.

        Returns:
            Formatted summary string.
        """
        closed = sum(1 for r in results if r.category == PARAM_VALIDATION.CLOSED)
        open_count = sum(1 for r in results if r.category == PARAM_VALIDATION.OPEN)
        invalid = sum(1 for r in results if r.category == PARAM_VALIDATION.INVALID)

        lines = [
            "",
            self._color("=" * 60, 'bold'),
            self._color("  MLPERF STORAGE BENCHMARK REPORT SUMMARY", 'bold'),
            self._color("=" * 60, 'bold'),
            "",
            f"  Total Runs: {len(results)}",
            f"  {self._color('CLOSED:', 'green')} {closed}",
            f"  {self._color('OPEN:', 'yellow')} {open_count}",
            f"  {self._color('INVALID:', 'red')} {invalid}",
            "",
            f"  Workload Submissions: {len(workload_results)}",
            "",
        ]

        return '\n'.join(lines)

    def generate(self, results: List['Result'], workload_results: Dict,
                 advanced_data: Optional[Dict] = None) -> str:
        """
        Generate complete table report.

        Args:
            results: List of Result objects.
            workload_results: Dictionary of workload results.
            advanced_data: Optional advanced data.

        Returns:
            Complete formatted report string.
        """
        sections = []

        # Summary
        sections.append(self.format_summary_table(results, workload_results))

        # Runs table
        sections.append("\n" + self._color("BENCHMARK RUNS", 'bold'))
        sections.append(self.format_runs_table(results))

        # Workloads table
        if workload_results:
            sections.append("\n" + self._color("WORKLOAD SUBMISSIONS", 'bold'))
            sections.append(self.format_workloads_table(workload_results))

        # Metrics table
        sections.append("\n" + self._color("PERFORMANCE METRICS", 'bold'))
        sections.append(self.format_metrics_table(results))

        # Advanced data
        if advanced_data:
            if 'param_ranges' in advanced_data:
                sections.append("\n" + self._color("PARAMETER RANGES", 'bold'))
                sections.append(self.format_parameter_ranges_table(advanced_data['param_ranges']))

            if 'cluster_info' in advanced_data:
                sections.append("\n" + self._color("CLUSTER INFORMATION", 'bold'))
                sections.append(self.format_cluster_info_table(advanced_data['cluster_info']))

        return '\n'.join(sections)

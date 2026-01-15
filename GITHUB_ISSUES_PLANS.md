# GitHub Issues Implementation Plans

This document contains detailed implementation plans for each open GitHub issue in the mlperf-storage repository.

---

## Issue #22: Improve Report Generation

**Status:** Open
**Author:** wvaske
**Created:** January 15, 2026
**Link:** https://github.com/wvaske/mlperf-storage/issues/22

### Summary

The issue requests enhanced reporting capabilities with multiple output formats and an advanced/debug mode for expanded data display.

### Requirements

1. **Multiple Output Formats:**
   - Formatted tables to standard output
   - Flat CSV file export
   - Excel documents with pre-built analysis and pivot tables

2. **Advanced/Debug Output Mode:**
   - Display range of input parameters for each workload
   - Include cluster configuration information

### Current State Analysis

The existing reporting system (`mlpstorage/reporting.py` and `mlpstorage/reporting/`) provides:
- Basic CSV export (flat format)
- JSON export
- Terminal output with color-coded validation badges
- Directory structure validation
- Validation message formatting

**Key Files:**
- `mlpstorage/reporting.py:419-431` - Basic CSV writer
- `mlpstorage/reporting.py:413-417` - JSON writer
- `mlpstorage/reporting/formatters.py` - Terminal formatting classes

### Implementation Plan

#### Phase 1: Refactor Report Generation Architecture

**Step 1.1: Create Report Format Interface**
- Location: `mlpstorage/reporting/formats/__init__.py`
- Create abstract base class `ReportFormat` with methods:
  - `generate(results: List[Result]) -> bytes`
  - `get_extension() -> str`
  - `get_content_type() -> str`

**Step 1.2: Implement Format Handlers**

1. **Table Format** (`mlpstorage/reporting/formats/table.py`):
   - Use `tabulate` or custom table renderer
   - Support column selection and sorting
   - Methods: `render_to_stdout()`, `render_to_string()`

2. **CSV Format** (`mlpstorage/reporting/formats/csv_format.py`):
   - Migrate existing CSV logic from `reporting.py:419-431`
   - Add configurable column selection
   - Support for multiple CSV files (runs, workloads, metrics)

3. **Excel Format** (`mlpstorage/reporting/formats/excel.py`):
   - Use `openpyxl` library
   - Create multiple worksheets:
     - Summary sheet with totals and category breakdown
     - Runs sheet with per-run data
     - Workloads sheet with submission-level data
     - Metrics sheet with performance data
   - Add pivot table configurations
   - Implement conditional formatting (color cells by category)
   - Add charts for AU visualization

**Step 1.3: Update CLI Arguments**
- Location: `mlpstorage/cli/utility_args.py`
- Add new arguments:
  ```
  --output-format: choices=['table', 'csv', 'excel', 'json', 'all']
  --output-file: Custom output file path
  --advanced-output: Enable extended data display
  --include-cluster-info: Include cluster configuration in output
  --include-param-ranges: Include parameter ranges in output
  ```

#### Phase 2: Implement Table Output

**Step 2.1: Create Table Renderer**
```python
# mlpstorage/reporting/formats/table.py

class TableFormatter:
    """Format results as tables for terminal display."""

    def format_runs_table(self, results: List[Result]) -> str:
        """Generate formatted table of runs."""

    def format_workloads_table(self, results: Dict) -> str:
        """Generate formatted table of workload submissions."""

    def format_metrics_table(self, results: List[Result]) -> str:
        """Generate formatted table of performance metrics."""
```

**Step 2.2: Integration**
- Update `ReportGenerator.print_results()` to use `TableFormatter`
- Add `--table-style` option (simple, grid, fancy_grid, github)

#### Phase 3: Implement Excel Output

**Step 3.1: Add openpyxl Dependency**
- Update `pyproject.toml` to include `openpyxl>=3.0.0`

**Step 3.2: Create Excel Generator**
```python
# mlpstorage/reporting/formats/excel.py

class ExcelReportGenerator:
    """Generate Excel reports with analysis and pivot tables."""

    def create_workbook(self, results_dir: str) -> Workbook:
        """Create Excel workbook with all data."""

    def add_summary_sheet(self, wb: Workbook, results: List[Result]):
        """Add summary sheet with counts and categories."""

    def add_runs_sheet(self, wb: Workbook, results: List[Result]):
        """Add detailed runs sheet."""

    def add_pivot_table(self, ws: Worksheet, data_range: str):
        """Add pivot table for data analysis."""

    def add_charts(self, ws: Worksheet, data: Dict):
        """Add performance charts."""
```

**Step 3.3: Pivot Table Features**
- Group by: benchmark_type, model, accelerator, category
- Metrics: AU percentage, throughput, run count
- Filter by: date range, category

#### Phase 4: Implement Advanced/Debug Output Mode

**Step 4.1: Create Advanced Data Collector**
```python
# mlpstorage/reporting/advanced_collector.py

class AdvancedDataCollector:
    """Collect extended data for advanced output mode."""

    def collect_param_ranges(self, runs: List[BenchmarkRun]) -> Dict:
        """Calculate min/max/avg for each numeric parameter."""

    def collect_cluster_details(self, runs: List[BenchmarkRun]) -> Dict:
        """Extract detailed cluster configuration."""

    def collect_timing_analysis(self, runs: List[BenchmarkRun]) -> Dict:
        """Analyze timing between runs and epochs."""
```

**Step 4.2: Parameter Range Analysis**
- For each workload, calculate:
  - `num_files_train`: min, max, average
  - `num_samples_per_file`: min, max, average
  - `read_threads`: values used
  - `prefetch_size`: values used
  - Throughput statistics (mean, std, min, max)

**Step 4.3: Cluster Information Display**
- Per-host information:
  - Hostname, CPU model, core count
  - Memory total/available
  - Storage device info (if collected)
  - Network interface info (if collected)
- Cluster consistency warnings
- OS and kernel version information

#### Phase 5: Update ReportGenerator

**Step 5.1: Refactor generate_reports()**
```python
def generate_reports(self, output_format: str = 'all',
                     advanced: bool = False) -> EXIT_CODE:
    """Generate reports in specified format(s)."""

    if advanced:
        self._collect_advanced_data()

    if output_format in ('table', 'all'):
        self._generate_table_output()
    if output_format in ('csv', 'all'):
        self._generate_csv_output()
    if output_format in ('excel', 'all'):
        self._generate_excel_output()
    if output_format in ('json', 'all'):
        self._generate_json_output()
```

**Step 5.2: Add Advanced Output Integration**
- When `--advanced-output` is set:
  - Include parameter ranges in all output formats
  - Include cluster details in all output formats
  - Add extra columns/sheets for advanced data

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mlpstorage/reporting/formats/__init__.py` | Create | Format registry and base class |
| `mlpstorage/reporting/formats/table.py` | Create | Table formatting |
| `mlpstorage/reporting/formats/excel.py` | Create | Excel generation |
| `mlpstorage/reporting/formats/csv_format.py` | Create | Enhanced CSV generation |
| `mlpstorage/reporting/advanced_collector.py` | Create | Advanced data collection |
| `mlpstorage/reporting.py` | Modify | Integrate new formats |
| `mlpstorage/cli/utility_args.py` | Modify | Add CLI arguments |
| `pyproject.toml` | Modify | Add openpyxl dependency |
| `tests/test_reporting_formats.py` | Create | Unit tests for formats |

### Dependencies to Add

```toml
[project.optional-dependencies]
excel = ["openpyxl>=3.0.0"]
full = ["openpyxl>=3.0.0", "tabulate>=0.9.0"]
```

### Testing Plan

1. Unit tests for each format generator
2. Integration tests with sample result directories
3. Verify Excel files open correctly in Excel/LibreOffice
4. Test advanced output mode with various cluster configurations
5. Performance test with large result sets (100+ runs)

---

## Issue #15: v2.0 Task List for Submission Checking

**Status:** Open
**Author:** wvaske
**Created:** July 11, 2025
**Link:** https://github.com/wvaske/mlperf-storage/issues/15

### Summary

Comprehensive task list for v2.0 submission validation including run count verification, AU targets, parameter consistency, configuration alignment, throughput calculations, and performance anomaly detection.

### Requirements Breakdown

#### Category 1: Basic Submission Validation
- [ ] Verify correct run count (5 runs for training)
- [ ] Verify AU is met for every run (not every epoch)
- [ ] Ensure parameters are consistent across runs
- [ ] Verify executed parameters match configuration files

#### Category 2: Data Validation
- [ ] Validate sample counts align with throughput calculations
- [ ] Confirm datagen logs reflect dataset composition
- [ ] Monitor timing intervals between test phases

#### Category 3: Storage/Checkpoint Validation
- [ ] Verify cache was flushed before checkpoint operations
- [ ] Confirm fsync usage for checkpoint writes
- [ ] Check checkpoint files in code

#### Category 4: Software Validation
- [ ] Validate DLIO version
- [ ] Validate mlpstorage version

#### Category 5: Performance Validation
- [ ] Maintain minimum 90% accelerator utilization
- [ ] Detect throughput anomalies per processing unit
- [ ] Identify training epoch gaps
- [ ] Monitor per-process checkpoint performance

#### Category 6: System Validation
- [ ] Cross-validate system description with collected CPU/memory metrics

### Current State Analysis

**Existing Implementation:**
- `mlpstorage/rules/run_checkers/training.py` - Basic parameter validation
- `mlpstorage/rules/submission_checkers/training.py` - Run count validation
- `mlpstorage/rules/verifier.py` - Verification orchestrator

**Placeholder Methods (need implementation):**
- `check_checkpoint_files_in_code()` - Line 175
- `check_num_epochs()` - Line 179
- `check_inter_test_times()` - Line 183
- `check_file_system_caching()` - Line 187

### Implementation Plan

#### Phase 1: Complete Run-Level Checkers

**Step 1.1: Implement AU Verification**
```python
# mlpstorage/rules/run_checkers/training.py

def check_accelerator_utilization(self) -> Optional[Issue]:
    """Verify AU meets minimum threshold for the run."""
    metrics = self.benchmark_run.metrics
    if not metrics:
        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message="No metrics available for AU verification",
            parameter="metrics"
        )

    au_percentage = metrics.get('au', 0)
    MIN_AU_THRESHOLD = 90.0

    if au_percentage < MIN_AU_THRESHOLD:
        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=f"AU {au_percentage:.1f}% below minimum {MIN_AU_THRESHOLD}%",
            parameter="accelerator_utilization",
            expected=f">= {MIN_AU_THRESHOLD}%",
            actual=f"{au_percentage:.1f}%"
        )
    return None
```

**Step 1.2: Implement Throughput Validation**
```python
# mlpstorage/rules/run_checkers/training.py

def check_throughput_calculations(self) -> Optional[Issue]:
    """Validate sample counts align with throughput calculations."""
    params = self.benchmark_run.parameters
    metrics = self.benchmark_run.metrics

    expected_samples = (
        params['dataset']['num_files_train'] *
        params['dataset']['num_samples_per_file']
    )

    reported_samples = metrics.get('samples_processed', 0)

    # Allow 1% tolerance for rounding
    if abs(expected_samples - reported_samples) / expected_samples > 0.01:
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="Sample count mismatch in throughput calculation",
            parameter="throughput_samples",
            expected=expected_samples,
            actual=reported_samples
        )
    return None
```

**Step 1.3: Implement Epoch Validation**
```python
def check_num_epochs(self) -> Optional[Issue]:
    """Verify epoch count and detect gaps."""
    metrics = self.benchmark_run.metrics
    params = self.benchmark_run.parameters

    expected_epochs = params.get('workflow', {}).get('epochs', 1)
    reported_epochs = metrics.get('epochs_completed', 0)

    if reported_epochs < expected_epochs:
        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=f"Incomplete epochs: {reported_epochs}/{expected_epochs}",
            parameter="epochs",
            expected=expected_epochs,
            actual=reported_epochs
        )
    return None
```

**Step 1.4: Implement Inter-Test Timing**
```python
def check_inter_test_times(self) -> Optional[Issue]:
    """Check timing intervals between test phases."""
    metrics = self.benchmark_run.metrics

    # Extract phase timings
    phase_times = metrics.get('phase_times', {})

    # Check for suspicious gaps (e.g., > 60s between phases)
    MAX_GAP_SECONDS = 60

    for phase, timing in phase_times.items():
        gap = timing.get('gap_before', 0)
        if gap > MAX_GAP_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Large gap ({gap}s) before {phase} phase",
                parameter="inter_test_timing",
                expected=f"< {MAX_GAP_SECONDS}s",
                actual=f"{gap}s"
            )
    return None
```

**Step 1.5: Implement Cache Validation**
```python
def check_file_system_caching(self) -> Optional[Issue]:
    """Verify cache was properly managed."""
    metadata = self.benchmark_run.parameters

    # Check if drop_caches was executed
    cache_dropped = metadata.get('cache_dropped', False)

    if not cache_dropped:
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="File system cache was not verified to be dropped",
            parameter="cache_management",
            suggestion="Ensure cache is dropped before benchmark runs"
        )
    return None
```

#### Phase 2: Complete Submission-Level Checkers

**Step 2.1: Parameter Consistency Checker**
```python
# mlpstorage/rules/submission_checkers/training.py

def check_parameter_consistency(self) -> List[Issue]:
    """Ensure parameters are consistent across all runs."""
    issues = []

    # Parameters that must be identical across runs
    CONSISTENT_PARAMS = [
        'dataset.num_files_train',
        'dataset.num_samples_per_file',
        'reader.read_threads',
        'workflow.epochs',
    ]

    for param in CONSISTENT_PARAMS:
        values = set()
        for run in self.benchmark_runs:
            value = self._get_nested_param(run.parameters, param)
            values.add(str(value))

        if len(values) > 1:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Inconsistent {param} across runs: {values}",
                parameter=param,
                expected="Consistent value across all runs",
                actual=str(values)
            ))

    return issues
```

**Step 2.2: Configuration Alignment Checker**
```python
def check_config_alignment(self) -> List[Issue]:
    """Verify executed parameters match configuration files."""
    issues = []

    for run in self.benchmark_runs:
        hydra_config = run.parameters  # From config.yaml
        override_params = run.override_parameters  # From overrides.yaml

        # Verify overrides were actually applied
        for param, expected_value in override_params.items():
            actual_value = self._get_nested_param(hydra_config, param)
            if str(actual_value) != str(expected_value):
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Override {param} not applied correctly",
                    parameter=param,
                    expected=expected_value,
                    actual=actual_value
                ))

    return issues
```

**Step 2.3: Warmup Run Support (addresses PR #16)**
```python
def check_num_runs(self) -> Optional[Issue]:
    """
    Require 5 runs for training benchmark closed submission.
    Supports 6 runs if 1 is marked as warmup.
    """
    num_runs = len(self.benchmark_runs)

    # Check for warmup runs
    warmup_runs = [r for r in self.benchmark_runs if r.is_warmup]
    valid_runs = [r for r in self.benchmark_runs if not r.is_warmup]

    if len(valid_runs) < self.REQUIRED_RUNS:
        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=f"Training submission requires {self.REQUIRED_RUNS} valid runs "
                    f"(found {len(valid_runs)}, {len(warmup_runs)} warmup)",
            parameter="num_runs",
            expected=self.REQUIRED_RUNS,
            actual=len(valid_runs)
        )

    # Note about warmup runs
    if warmup_runs:
        return Issue(
            validation=PARAM_VALIDATION.CLOSED,
            message=f"Submission has {len(valid_runs)} valid runs + "
                    f"{len(warmup_runs)} warmup run(s)",
            parameter="num_runs",
            expected=f">= {self.REQUIRED_RUNS}",
            actual=len(valid_runs)
        )

    return Issue(
        validation=PARAM_VALIDATION.CLOSED,
        message=f"Training submission has required {self.REQUIRED_RUNS} runs",
        parameter="num_runs",
        expected=self.REQUIRED_RUNS,
        actual=num_runs
    )
```

#### Phase 3: Performance Anomaly Detection

**Step 3.1: Create Performance Analyzer**
```python
# mlpstorage/rules/performance_analyzer.py

class PerformanceAnalyzer:
    """Detect performance anomalies in benchmark runs."""

    def __init__(self, benchmark_runs: List[BenchmarkRun], logger=None):
        self.benchmark_runs = benchmark_runs
        self.logger = logger

    def analyze_throughput_variance(self) -> List[Issue]:
        """Detect throughput anomalies across runs."""
        throughputs = [r.metrics.get('throughput', 0) for r in self.benchmark_runs]

        mean_throughput = sum(throughputs) / len(throughputs)
        variance = sum((t - mean_throughput) ** 2 for t in throughputs) / len(throughputs)
        std_dev = variance ** 0.5

        issues = []
        for i, run in enumerate(self.benchmark_runs):
            throughput = throughputs[i]
            z_score = (throughput - mean_throughput) / std_dev if std_dev > 0 else 0

            if abs(z_score) > 2:  # More than 2 standard deviations
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Throughput anomaly detected in run {run.run_id}",
                    parameter="throughput_variance",
                    expected=f"Within 2 std dev of mean ({mean_throughput:.2f})",
                    actual=f"{throughput:.2f} (z-score: {z_score:.2f})"
                ))

        return issues

    def detect_epoch_gaps(self) -> List[Issue]:
        """Identify gaps in training epochs."""
        issues = []

        for run in self.benchmark_runs:
            epoch_times = run.metrics.get('epoch_times', [])

            for i in range(1, len(epoch_times)):
                gap = epoch_times[i]['start'] - epoch_times[i-1]['end']

                # Flag gaps > 10% of average epoch time
                avg_epoch_time = sum(e['duration'] for e in epoch_times) / len(epoch_times)
                if gap > avg_epoch_time * 0.1:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.OPEN,
                        message=f"Gap between epochs {i-1} and {i} in run {run.run_id}",
                        parameter="epoch_gap",
                        expected=f"< {avg_epoch_time * 0.1:.2f}s",
                        actual=f"{gap:.2f}s"
                    ))

        return issues

    def analyze_per_process_performance(self) -> List[Issue]:
        """Monitor per-process checkpoint performance."""
        issues = []

        for run in self.benchmark_runs:
            per_process_metrics = run.metrics.get('per_process', {})

            throughputs = [m.get('throughput', 0) for m in per_process_metrics.values()]
            if not throughputs:
                continue

            mean_throughput = sum(throughputs) / len(throughputs)

            for proc_id, metrics in per_process_metrics.items():
                proc_throughput = metrics.get('throughput', 0)
                if mean_throughput > 0 and proc_throughput < mean_throughput * 0.8:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.OPEN,
                        message=f"Process {proc_id} underperforming in run {run.run_id}",
                        parameter="per_process_throughput",
                        expected=f">= 80% of mean ({mean_throughput * 0.8:.2f})",
                        actual=f"{proc_throughput:.2f}"
                    ))

        return issues
```

#### Phase 4: System Description Validation

**Step 4.1: Implement System Cross-Validation**
```python
# mlpstorage/rules/submission_checkers/system_validator.py

class SystemDescriptionValidator:
    """Cross-validate system description with collected metrics."""

    def validate_cpu_info(self, declared: Dict, collected: ClusterInformation) -> List[Issue]:
        """Verify declared CPU info matches collected data."""
        issues = []

        for host in collected.host_info_list:
            if host.cpu:
                declared_cores = declared.get('cpu_cores', 0)
                if declared_cores != host.cpu.num_cores:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.OPEN,
                        message=f"CPU core mismatch on {host.hostname}",
                        parameter="system.cpu_cores",
                        expected=declared_cores,
                        actual=host.cpu.num_cores
                    ))

        return issues

    def validate_memory_info(self, declared: Dict, collected: ClusterInformation) -> List[Issue]:
        """Verify declared memory matches collected data."""
        issues = []

        declared_memory_gb = declared.get('total_memory_gb', 0)
        collected_memory_gb = collected.total_memory_bytes / (1024 ** 3)

        # Allow 5% tolerance for memory reporting differences
        if abs(declared_memory_gb - collected_memory_gb) / declared_memory_gb > 0.05:
            issues.append(Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="Total memory mismatch",
                parameter="system.total_memory",
                expected=f"{declared_memory_gb:.1f} GB",
                actual=f"{collected_memory_gb:.1f} GB"
            ))

        return issues
```

#### Phase 5: Version Validation

**Step 5.1: Add Version Checkers**
```python
# mlpstorage/rules/run_checkers/base.py

def check_dlio_version(self) -> Optional[Issue]:
    """Validate DLIO benchmark version."""
    metadata = self.benchmark_run.parameters
    dlio_version = metadata.get('dlio_version', 'unknown')

    REQUIRED_DLIO_VERSION = "1.0.0"  # Update as needed

    if dlio_version == 'unknown':
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="DLIO version not recorded in metadata",
            parameter="dlio_version",
            suggestion="Ensure DLIO version is captured during benchmark"
        )

    # Version comparison logic
    if not self._version_satisfies(dlio_version, REQUIRED_DLIO_VERSION):
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message=f"DLIO version {dlio_version} may not meet requirements",
            parameter="dlio_version",
            expected=f">= {REQUIRED_DLIO_VERSION}",
            actual=dlio_version
        )

    return None

def check_mlpstorage_version(self) -> Optional[Issue]:
    """Validate mlpstorage version."""
    metadata = self.benchmark_run.parameters
    mlps_version = metadata.get('mlpstorage_version', 'unknown')

    # Record version for reference
    return Issue(
        validation=PARAM_VALIDATION.CLOSED,
        message=f"MLPerf Storage version: {mlps_version}",
        parameter="mlpstorage_version",
        actual=mlps_version
    )
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mlpstorage/rules/run_checkers/training.py` | Modify | Add AU, throughput, epoch, timing checks |
| `mlpstorage/rules/run_checkers/base.py` | Modify | Add version validators |
| `mlpstorage/rules/submission_checkers/training.py` | Modify | Add consistency, config, warmup checks |
| `mlpstorage/rules/performance_analyzer.py` | Create | Performance anomaly detection |
| `mlpstorage/rules/submission_checkers/system_validator.py` | Create | System description validation |
| `mlpstorage/rules/models.py` | Modify | Add warmup flag to BenchmarkRun |
| `tests/test_submission_checkers.py` | Create | Unit tests |
| `tests/test_performance_analyzer.py` | Create | Unit tests |

### Testing Plan

1. Create test fixtures with various validation scenarios
2. Test edge cases (partial data, missing metrics)
3. Test performance with large submission sets
4. Verify warmup run handling matches PR #16 intent
5. Integration tests with real benchmark outputs

---

## PR #16: Allow Training Submissions with 6 Runs (1 Warmup + 5 Runs)

**Status:** Open (awaiting merge)
**Author:** OMichaud0
**Created:** July 17, 2025
**Target Branch:** v2.0-submission-checker
**Link:** https://github.com/wvaske/mlperf-storage/pulls/16

### Summary

This PR adds the ability to validate training submissions with 5 valid runs plus 1 warmup run (6 total).

### Analysis

The PR modifies the submission checker to:
1. Accept 6 runs if 1 is marked as warmup
2. Only count non-warmup runs toward the 5-run requirement

### Recommendation

The functionality from PR #16 should be incorporated into the Issue #15 implementation plan, specifically in the `check_num_runs()` method refactoring (Phase 2, Step 2.3 above).

**Integration approach:**
1. Review and merge PR #16 first, or
2. Implement the warmup run logic as part of the Issue #15 work

Since PR #16 targets `v2.0-submission-checker` branch, the implementation in Issue #15 Plan Phase 2 already accounts for this feature.

---

## Implementation Priority

### Recommended Order

1. **Issue #15 - Submission Checking** (High Priority)
   - Fundamental to benchmark validity
   - Enables proper v2.0 submissions
   - PR #16 should be merged/incorporated here

2. **Issue #22 - Report Generation** (Medium Priority)
   - User experience improvement
   - Can be developed in parallel after Phase 1 of Issue #15

### Dependencies

```
Issue #15 Phase 1-2 ──► PR #16 Integration ──► Issue #15 Phase 3-5
                              │
                              └──► Issue #22 (parallel development)
```

---

## Summary

| Issue | Phases | New Files | Modified Files | Estimated Complexity |
|-------|--------|-----------|----------------|---------------------|
| #22 | 5 | 5 | 4 | Medium |
| #15 | 5 | 3 | 4 | High |
| PR #16 | - | 0 | 1 | Low (merge only) |


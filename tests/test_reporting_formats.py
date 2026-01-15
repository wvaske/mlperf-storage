"""
Tests for the reporting format handlers.

Tests cover:
- Table format output generation
- CSV format output generation
- JSON format output generation
- Advanced data collection
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES
from mlpstorage.reporting.formats.table import TableFormat
from mlpstorage.reporting.formats.csv_format import CSVFormat
from mlpstorage.reporting.formats.json_format import JSONFormat
from mlpstorage.reporting.formats import FormatConfig
from mlpstorage.reporting.advanced_collector import AdvancedDataCollector, collect_advanced_data


# Mock Result class for testing
@dataclass
class MockResult:
    """Mock Result class for testing."""
    multi: bool
    benchmark_type: Any
    benchmark_command: str
    benchmark_model: str
    benchmark_run: Any
    issues: List
    category: PARAM_VALIDATION
    metrics: Dict[str, Any]


# Mock BenchmarkRun class for testing
@dataclass
class MockBenchmarkRun:
    """Mock BenchmarkRun class for testing."""
    run_id: str
    benchmark_type: Any
    model: str
    command: str
    num_processes: int
    accelerator: str
    run_datetime: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    system_info: Any = None


@pytest.fixture
def mock_results():
    """Create mock results for testing."""
    run1 = MockBenchmarkRun(
        run_id="training_unet3d_run_20250115_120000",
        benchmark_type=BENCHMARK_TYPES.training,
        model="unet3d",
        command="run",
        num_processes=8,
        accelerator="h100",
        run_datetime="20250115_120000",
        parameters={"dataset": {"num_files_train": 1000}},
        metrics={"au": 95.5, "throughput": 1234.56}
    )

    run2 = MockBenchmarkRun(
        run_id="training_unet3d_run_20250115_130000",
        benchmark_type=BENCHMARK_TYPES.training,
        model="unet3d",
        command="run",
        num_processes=8,
        accelerator="h100",
        run_datetime="20250115_130000",
        parameters={"dataset": {"num_files_train": 1000}},
        metrics={"au": 92.3, "throughput": 1198.45}
    )

    results = [
        MockResult(
            multi=False,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command="run",
            benchmark_model="unet3d",
            benchmark_run=run1,
            issues=[],
            category=PARAM_VALIDATION.CLOSED,
            metrics={"au": 95.5, "throughput": 1234.56}
        ),
        MockResult(
            multi=False,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command="run",
            benchmark_model="unet3d",
            benchmark_run=run2,
            issues=[],
            category=PARAM_VALIDATION.OPEN,
            metrics={"au": 92.3, "throughput": 1198.45}
        ),
    ]
    return results


@pytest.fixture
def mock_workload_results(mock_results):
    """Create mock workload results for testing."""
    return {
        ("unet3d", "h100"): MockResult(
            multi=True,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command="run",
            benchmark_model="unet3d",
            benchmark_run=[mock_results[0].benchmark_run, mock_results[1].benchmark_run],
            issues=[],
            category=PARAM_VALIDATION.CLOSED,
            metrics={}
        )
    }


class TestTableFormat:
    """Tests for TableFormat class."""

    def test_init_default(self):
        """Test default initialization."""
        formatter = TableFormat()
        assert formatter.use_colors is True
        assert formatter.max_width == 120

    def test_init_custom(self):
        """Test custom initialization."""
        formatter = TableFormat(style='grid', use_colors=False, max_width=80)
        assert formatter.use_colors is False
        assert formatter.max_width == 80

    def test_format_runs_table_empty(self):
        """Test formatting empty results."""
        formatter = TableFormat()
        output = formatter.format_runs_table([])
        assert "No runs to display" in output

    def test_format_runs_table(self, mock_results):
        """Test formatting runs table."""
        formatter = TableFormat(use_colors=False)
        output = formatter.format_runs_table(mock_results)

        assert "Run ID" in output
        assert "Type" in output
        assert "Model" in output
        assert "Category" in output
        assert "unet3d" in output
        assert "training" in output

    def test_format_workloads_table(self, mock_workload_results):
        """Test formatting workloads table."""
        formatter = TableFormat(use_colors=False)
        output = formatter.format_workloads_table(mock_workload_results)

        assert "Model" in output
        assert "Accelerator" in output
        assert "unet3d" in output
        assert "h100" in output

    def test_format_metrics_table(self, mock_results):
        """Test formatting metrics table."""
        formatter = TableFormat(use_colors=False)
        output = formatter.format_metrics_table(mock_results)

        assert "Run ID" in output

    def test_format_summary_table(self, mock_results, mock_workload_results):
        """Test formatting summary table."""
        formatter = TableFormat(use_colors=False)
        output = formatter.format_summary_table(mock_results, mock_workload_results)

        assert "MLPERF STORAGE BENCHMARK REPORT SUMMARY" in output
        assert "Total Runs:" in output

    def test_generate_complete_report(self, mock_results, mock_workload_results):
        """Test generating complete report."""
        formatter = TableFormat(use_colors=False)
        output = formatter.generate(mock_results, mock_workload_results)

        assert "BENCHMARK RUNS" in output
        assert "WORKLOAD SUBMISSIONS" in output
        assert "PERFORMANCE METRICS" in output


class TestCSVFormat:
    """Tests for CSVFormat class."""

    def test_init_default(self):
        """Test default initialization."""
        config = FormatConfig()
        formatter = CSVFormat(config=config)
        assert formatter.name == "csv"
        assert formatter.extension == "csv"

    def test_generate_runs_csv_empty(self):
        """Test generating CSV with empty results."""
        formatter = CSVFormat()
        output = formatter.generate_runs_csv([])
        assert output == ""

    def test_generate_runs_csv(self, mock_results):
        """Test generating runs CSV."""
        formatter = CSVFormat()
        output = formatter.generate_runs_csv(mock_results)

        assert "run_id" in output
        assert "benchmark_type" in output
        assert "model" in output
        assert "category" in output
        assert "training" in output
        assert "unet3d" in output

    def test_generate_workloads_csv(self, mock_workload_results):
        """Test generating workloads CSV."""
        formatter = CSVFormat()
        output = formatter.generate_workloads_csv(mock_workload_results)

        assert "model" in output
        assert "accelerator" in output
        assert "unet3d" in output
        assert "h100" in output

    def test_generate_metrics_csv(self, mock_results):
        """Test generating metrics CSV."""
        formatter = CSVFormat()
        output = formatter.generate_metrics_csv(mock_results)

        assert "run_id" in output
        assert "model" in output

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        formatter = CSVFormat()
        nested = {"a": {"b": {"c": 1}}, "d": 2}
        flat = formatter._flatten_dict(nested)

        assert flat["a.b.c"] == 1
        assert flat["d"] == 2


class TestJSONFormat:
    """Tests for JSONFormat class."""

    def test_init_default(self):
        """Test default initialization."""
        formatter = JSONFormat()
        assert formatter.name == "json"
        assert formatter.extension == "json"
        assert formatter.indent == 2

    def test_generate_empty(self):
        """Test generating JSON with empty results."""
        formatter = JSONFormat()
        output = formatter.generate([], {})
        data = json.loads(output.decode('utf-8'))

        assert "summary" in data
        assert data["summary"]["total_runs"] == 0

    def test_generate_with_results(self, mock_results, mock_workload_results):
        """Test generating JSON with results."""
        formatter = JSONFormat()
        output = formatter.generate(mock_results, mock_workload_results)
        data = json.loads(output.decode('utf-8'))

        assert data["summary"]["total_runs"] == 2
        assert data["summary"]["closed_runs"] == 1
        assert data["summary"]["open_runs"] == 1
        assert len(data["runs"]) == 2
        assert len(data["workloads"]) == 1

    def test_generate_with_advanced_data(self, mock_results, mock_workload_results):
        """Test generating JSON with advanced data."""
        formatter = JSONFormat()
        advanced_data = {
            "param_ranges": {"test_workload": {"param1": {"min": 1, "max": 10}}},
            "cluster_info": {"num_hosts": 4, "total_memory_bytes": 1024**4}
        }
        output = formatter.generate(mock_results, mock_workload_results, advanced_data)
        data = json.loads(output.decode('utf-8'))

        assert "parameter_ranges" in data
        assert "cluster_info" in data


class TestAdvancedDataCollector:
    """Tests for AdvancedDataCollector class."""

    def test_init(self):
        """Test initialization."""
        collector = AdvancedDataCollector()
        assert collector.logger is None

    def test_collect_param_ranges_empty(self):
        """Test collecting param ranges with no runs."""
        collector = AdvancedDataCollector()
        result = collector.collect_param_ranges([])
        assert result == {}

    def test_collect_param_ranges(self, mock_results):
        """Test collecting param ranges."""
        collector = AdvancedDataCollector()
        runs = [r.benchmark_run for r in mock_results]
        result = collector.collect_param_ranges(runs)

        # Should have entries for the workload
        assert len(result) > 0

    def test_collect_cluster_details_empty(self):
        """Test collecting cluster details with no runs."""
        collector = AdvancedDataCollector()
        result = collector.collect_cluster_details([])
        assert result is None

    def test_collect_all(self, mock_results):
        """Test collecting all advanced data."""
        collector = AdvancedDataCollector()
        runs = [r.benchmark_run for r in mock_results]
        result = collector.collect_all(runs)

        assert "param_ranges" in result
        assert "cluster_info" in result


class TestConvenienceFunction:
    """Tests for collect_advanced_data convenience function."""

    def test_collect_advanced_data_empty(self):
        """Test convenience function with empty results."""
        result = collect_advanced_data([])
        assert "param_ranges" in result
        assert "cluster_info" in result

    def test_collect_advanced_data(self, mock_results):
        """Test convenience function with results."""
        result = collect_advanced_data(mock_results)
        assert "param_ranges" in result


class TestFormatConfig:
    """Tests for FormatConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = FormatConfig()
        assert config.output_path is None
        assert config.include_advanced is False
        assert config.include_cluster_info is False
        assert config.include_param_ranges is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = FormatConfig(
            output_path="/tmp/report.csv",
            include_advanced=True,
            include_cluster_info=True,
            include_param_ranges=True
        )
        assert config.output_path == "/tmp/report.csv"
        assert config.include_advanced is True
        assert config.include_cluster_info is True
        assert config.include_param_ranges is True

"""
Report format handlers for MLPerf Storage benchmark results.

This package provides multiple output formats for benchmark reports:
- Table: Formatted tables for terminal display
- CSV: Flat CSV files for data analysis
- Excel: Excel workbooks with analysis and pivot tables
- JSON: JSON format for programmatic access

Usage:
    from mlpstorage.reporting.formats import (
        ReportFormat,
        TableFormat,
        CSVFormat,
        ExcelFormat,
        JSONFormat,
        FormatRegistry,
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mlpstorage.reporting import Result


@dataclass
class FormatConfig:
    """Configuration for report format generation."""
    output_path: Optional[str] = None
    include_advanced: bool = False
    include_cluster_info: bool = False
    include_param_ranges: bool = False


class ReportFormat(ABC):
    """Abstract base class for report format handlers."""

    def __init__(self, config: Optional[FormatConfig] = None):
        """
        Initialize the format handler.

        Args:
            config: Optional configuration for the format.
        """
        self.config = config or FormatConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the format name."""
        pass

    @property
    @abstractmethod
    def extension(self) -> str:
        """Return the file extension for this format."""
        pass

    @property
    def content_type(self) -> str:
        """Return the MIME content type for this format."""
        return "application/octet-stream"

    @abstractmethod
    def generate(self, results: List['Result'], workload_results: Dict,
                 advanced_data: Optional[Dict] = None) -> bytes:
        """
        Generate the report in this format.

        Args:
            results: List of Result objects from single runs.
            workload_results: Dictionary of workload-level results.
            advanced_data: Optional advanced data (param ranges, cluster info).

        Returns:
            Bytes containing the formatted report.
        """
        pass

    def generate_to_file(self, results: List['Result'], workload_results: Dict,
                         output_path: str, advanced_data: Optional[Dict] = None) -> str:
        """
        Generate the report and write to a file.

        Args:
            results: List of Result objects from single runs.
            workload_results: Dictionary of workload-level results.
            output_path: Path to write the output file.
            advanced_data: Optional advanced data.

        Returns:
            Path to the written file.
        """
        content = self.generate(results, workload_results, advanced_data)

        mode = 'wb' if isinstance(content, bytes) else 'w'
        with open(output_path, mode) as f:
            f.write(content)

        return output_path


class FormatRegistry:
    """Registry for available report formats."""

    _formats: Dict[str, type] = {}

    @classmethod
    def register(cls, format_class: type) -> type:
        """
        Register a format class.

        Can be used as a decorator:
            @FormatRegistry.register
            class MyFormat(ReportFormat):
                ...
        """
        instance = format_class()
        cls._formats[instance.name] = format_class
        return format_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a format class by name."""
        return cls._formats.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, type]:
        """Get all registered formats."""
        return cls._formats.copy()

    @classmethod
    def available_formats(cls) -> List[str]:
        """Get list of available format names."""
        return list(cls._formats.keys())


# Import format implementations to register them
from mlpstorage.reporting.formats.table import TableFormat
from mlpstorage.reporting.formats.csv_format import CSVFormat
from mlpstorage.reporting.formats.excel import ExcelFormat
from mlpstorage.reporting.formats.json_format import JSONFormat

__all__ = [
    'ReportFormat',
    'FormatConfig',
    'FormatRegistry',
    'TableFormat',
    'CSVFormat',
    'ExcelFormat',
    'JSONFormat',
]

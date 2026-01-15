"""
Reporting system for MLPerf Storage benchmark results.

This package provides:
- Directory structure validation
- Validation message formatting
- OPEN vs CLOSED submission messaging
- Report generation utilities
- Multiple output formats (table, CSV, Excel, JSON)
- Advanced data collection (parameter ranges, cluster info)

Modules:
    - directory_validator: Validate results directory structure
    - formatters: Format validation messages for display
    - formats: Output format handlers (table, CSV, Excel, JSON)
    - advanced_collector: Advanced data collection for reports

Usage:
    from mlpstorage.reporting import (
        ResultsDirectoryValidator,
        ValidationMessageFormatter,
        ClosedRequirementsFormatter,
        ReportSummaryFormatter,
    )

    # For format-specific output:
    from mlpstorage.reporting.formats import (
        TableFormat,
        CSVFormat,
        ExcelFormat,
        JSONFormat,
    )

    # For advanced data collection:
    from mlpstorage.reporting.advanced_collector import (
        AdvancedDataCollector,
        collect_advanced_data,
    )
"""

from mlpstorage.reporting.directory_validator import (
    ResultsDirectoryValidator,
    DirectoryValidationError,
    DirectoryValidationResult,
)

from mlpstorage.reporting.formatters import (
    ValidationMessageFormatter,
    ClosedRequirementsFormatter,
    ReportSummaryFormatter,
)

from mlpstorage.reporting.advanced_collector import (
    AdvancedDataCollector,
    collect_advanced_data,
)

__all__ = [
    # Directory validation
    'ResultsDirectoryValidator',
    'DirectoryValidationError',
    'DirectoryValidationResult',
    # Formatters
    'ValidationMessageFormatter',
    'ClosedRequirementsFormatter',
    'ReportSummaryFormatter',
    # Advanced data collection
    'AdvancedDataCollector',
    'collect_advanced_data',
]

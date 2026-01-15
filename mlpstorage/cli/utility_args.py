"""
Utility CLI argument builders for non-benchmark commands.

This module defines the CLI arguments for utility commands like
reports and history.
"""

from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
)


# Output format choices
OUTPUT_FORMATS = ['table', 'csv', 'excel', 'json', 'all']

# Help messages for report arguments
REPORT_HELP = {
    'output_format': (
        "Output format for the report. Options: "
        "'table' (formatted terminal output), "
        "'csv' (flat CSV files), "
        "'excel' (Excel workbook with analysis), "
        "'json' (JSON format), "
        "'all' (generate all formats). Default: table"
    ),
    'output_file': (
        "Custom output file path. If not specified, files are written "
        "to the results directory with auto-generated names."
    ),
    'advanced_output': (
        "Enable advanced output mode. Includes extended data such as "
        "parameter ranges across runs and detailed cluster configuration."
    ),
    'include_cluster_info': (
        "Include detailed cluster configuration in the report. "
        "Shows per-host CPU, memory, and system information."
    ),
    'include_param_ranges': (
        "Include parameter range analysis in the report. "
        "Shows min/max/avg values for numeric parameters across runs."
    ),
    'table_style': (
        "Table style for terminal output. Options: "
        "'simple' (basic borders), "
        "'grid' (full grid), "
        "'minimal' (no borders). Default: simple"
    ),
    'no_colors': (
        "Disable terminal colors in table output."
    ),
}


def add_reports_arguments(parser):
    """Add reports command arguments to the parser.

    Args:
        parser: Argparse subparser for the reports command.
    """
    reports_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Sub-commands"
    )
    parser.required = True

    reportgen = reports_subparsers.add_parser(
        'reportgen',
        help=HELP_MESSAGES['reportgen']
    )

    reportgen.add_argument(
        '--output-dir',
        type=str,
        help=HELP_MESSAGES['output_dir']
    )

    # Output format options
    reportgen.add_argument(
        '--output-format', '-f',
        type=str,
        choices=OUTPUT_FORMATS,
        default='table',
        help=REPORT_HELP['output_format']
    )

    reportgen.add_argument(
        '--output-file', '-o',
        type=str,
        help=REPORT_HELP['output_file']
    )

    # Advanced output options
    reportgen.add_argument(
        '--advanced-output', '--advanced',
        action='store_true',
        default=False,
        help=REPORT_HELP['advanced_output']
    )

    reportgen.add_argument(
        '--include-cluster-info',
        action='store_true',
        default=False,
        help=REPORT_HELP['include_cluster_info']
    )

    reportgen.add_argument(
        '--include-param-ranges',
        action='store_true',
        default=False,
        help=REPORT_HELP['include_param_ranges']
    )

    # Table formatting options
    reportgen.add_argument(
        '--table-style',
        type=str,
        choices=['simple', 'grid', 'minimal'],
        default='simple',
        help=REPORT_HELP['table_style']
    )

    reportgen.add_argument(
        '--no-colors',
        action='store_true',
        default=False,
        help=REPORT_HELP['no_colors']
    )

    add_universal_arguments(reportgen)


def add_history_arguments(parser):
    """Add history command arguments to the parser.

    Args:
        parser: Argparse subparser for the history command.
    """
    history_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Sub-commands"
    )
    parser.required = True

    history = history_subparsers.add_parser(
        'show',
        help="Show command history"
    )
    history.add_argument(
        '--limit', '-n',
        type=int,
        help="Limit to the N most recent commands"
    )
    history.add_argument(
        '--id', '-i',
        type=int,
        help="Show a specific command by ID"
    )

    rerun = history_subparsers.add_parser(
        'rerun',
        help="Re-run a command from history"
    )
    rerun.add_argument(
        'rerun_id',
        type=int,
        help="ID of the command to re-run"
    )

    for _parser in [history, rerun]:
        add_universal_arguments(_parser)

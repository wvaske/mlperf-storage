import abc
import enum
import json
import os
import yaml

from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint, pformat
from typing import List, Dict, Any, Optional, Tuple

from mlpstorage.config import (MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, BENCHMARK_TYPES,
                               DATETIME_STR, LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, HYDRA_OUTPUT_SUBDIR, UNET)
from mlpstorage.mlps_logging import setup_logging
from mlpstorage.utils import is_valid_datetime_format
from mlpstorage.cluster_collector import (
    HostDiskInfo,
    HostNetworkInfo,
    HostSystemInfo,
    summarize_cpuinfo,
)


class RuleState(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    INVALID = "invalid"


@dataclass
class Issue:
    validation: PARAM_VALIDATION
    message: str
    parameter: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    severity: str = "error"
    
    def __str__(self):
        result = f"[{self.validation.value.upper()}] {self.message}"
        if self.parameter:
            result += f" (Parameter: {self.parameter}"
            if self.expected is not None and self.actual is not None:
                result += f", Expected: {self.expected}, Actual: {self.actual}"
            result += ")"
        return result


@dataclass
class RunID:
    program: str
    command: str
    model: str
    run_datetime: str

    def __str__(self):
        id_str = self.program
        if self.command:
            id_str += f"_{self.command}"
        if self.model:
            id_str += f"_{self.model}"
        id_str += f"_{self.run_datetime}"
        return id_str


@dataclass
class ProcessedRun:
    run_id: RunID
    benchmark_type: str
    run_parameters: Dict[str, Any]
    run_metrics: Dict[str, Any]
    issues: List[Issue] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if the run is valid (no issues with INVALID validation)"""
        return not any(issue.validation == PARAM_VALIDATION.INVALID for issue in self.issues)
    
    def is_closed(self) -> bool:
        """Check if the run is valid for closed submission"""
        if not self.is_valid():
            return False
        return all(issue.validation != PARAM_VALIDATION.OPEN for issue in self.issues)


@dataclass
class HostMemoryInfo:
    """Detailed memory information for a host"""
    total: int  # Total physical memory in bytes
    available: Optional[int]  # Memory available for allocation
    used: Optional[int]  # Memory currently in use
    free: Optional[int]  # Memory not being used
    active: Optional[int]  # Memory actively used
    inactive: Optional[int]  # Memory marked as inactive
    buffers: Optional[int]  # Memory used for buffers
    cached: Optional[int]  # Memory used for caching
    shared: Optional[int]  # Memory shared between processes

    @classmethod
    def from_psutil_dict(cls, data: Dict[str, int]) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from a dictionary"""
        return cls(
            total=data.get('total', 0),
            available=data.get('available', 0),
            used=data.get('used', 0),
            free=data.get('free', 0),
            active=data.get('active', 0),
            inactive=data.get('inactive', 0),
            buffers=data.get('buffers', 0),
            cached=data.get('cached', 0),
            shared=data.get('shared', 0)
        )

    @classmethod
    def from_proc_meminfo_dict(cls, data: Dict[str, Any]) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from a parsed /proc/meminfo dictionary.

        Args:
            data: Dictionary with keys like 'MemTotal', 'MemFree', etc.
                  Values should be integers in kB (as returned by parse_proc_meminfo).

        Returns:
            HostMemoryInfo instance with values converted to bytes.
        """
        def get_bytes(key: str, default: int = 0) -> int:
            """Get value in bytes (input is in kB)."""
            val = data.get(key, default)
            if isinstance(val, (int, float)):
                return int(val) * 1024
            # Handle string values with units (legacy support)
            if isinstance(val, str):
                try:
                    return int(val.split()[0]) * 1024
                except (ValueError, IndexError):
                    return default
            return default

        return cls(
            total=get_bytes('MemTotal'),
            available=get_bytes('MemAvailable'),
            used=get_bytes('MemTotal') - get_bytes('MemFree') - get_bytes('Buffers') - get_bytes('Cached'),
            free=get_bytes('MemFree'),
            active=get_bytes('Active'),
            inactive=get_bytes('Inactive'),
            buffers=get_bytes('Buffers'),
            cached=get_bytes('Cached'),
            shared=get_bytes('Shmem'),
        )

    @classmethod
    def from_total_mem_int(cls, total_mem_int: int) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from total memory in bytes"""
        return cls(
            total=total_mem_int,
            available=None,
            used=None,
            free=None,
            active=None,
            inactive=None,
            buffers=None,
            cached=None,
            shared=None
        )


@dataclass
class HostCPUInfo:
    """CPU information for a host"""
    num_cores: int = 0  # Number of physical CPU cores
    num_logical_cores: int = 0  # Number of logical CPU cores (with hyperthreading)
    model: str = ""  # CPU model name
    architecture: str = ""  # CPU architecture

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostCPUInfo':
        """Create a HostCPUInfo instance from a dictionary"""
        return cls(
            num_cores=data.get('num_cores', 0),
            num_logical_cores=data.get('num_logical_cores', 0),
            model=data.get('model', ""),
            architecture=data.get('architecture', ""),
        )


@dataclass
class HostInfo:
    """Information about a single host in the system.

    Attributes:
        hostname: The hostname or IP address of the host.
        memory: Memory information from /proc/meminfo.
        cpu: CPU information from /proc/cpuinfo.
        disks: List of disk statistics from /proc/diskstats.
        network: List of network interface statistics from /proc/net/dev.
        system: Extended system information (kernel, OS, uptime, load).
        collection_timestamp: ISO timestamp when the data was collected.
    """
    hostname: str
    memory: HostMemoryInfo = field(default_factory=HostMemoryInfo)
    cpu: Optional[HostCPUInfo] = None
    disks: Optional[List[HostDiskInfo]] = None
    network: Optional[List[HostNetworkInfo]] = None
    system: Optional[HostSystemInfo] = None
    collection_timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, hostname: str, data: Dict[str, Any]) -> 'HostInfo':
        """Create a HostInfo instance from a dictionary"""
        memory_info = data.get('memory_info', {})
        cpu_info = data.get('cpu_info', {})

        # Determine which memory info constructor to use based on the data structure
        if isinstance(memory_info, dict):
            # Check if it looks like psutil data
            if 'total' in memory_info and isinstance(memory_info['total'], int):
                memory = HostMemoryInfo.from_psutil_dict(memory_info)
            # Check if it looks like proc_meminfo data
            elif 'MemTotal' in memory_info:
                memory = HostMemoryInfo.from_proc_meminfo_dict(memory_info)
            else:
                # Default to empty memory info if we can't determine the format
                memory = HostMemoryInfo()
        else:
            memory = HostMemoryInfo()

        # Handle the case where cpu_info is None or empty
        cpu = None
        if cpu_info:
            cpu = HostCPUInfo.from_dict(cpu_info)

        return cls(
            hostname=hostname,
            memory=memory,
            cpu=cpu,
        )

    @classmethod
    def from_collected_data(cls, data: Dict[str, Any]) -> 'HostInfo':
        """Create a HostInfo instance from MPI-collected data.

        Args:
            data: Dictionary from collect_local_system_info() containing:
                - hostname: str
                - meminfo: Dict from /proc/meminfo
                - cpuinfo: List[Dict] from /proc/cpuinfo
                - diskstats: List[Dict] disk info
                - netdev: List[Dict] network info
                - version: str kernel version
                - loadavg: Dict with load averages
                - uptime_seconds: float
                - os_release: Dict OS info
                - collection_timestamp: str

        Returns:
            HostInfo instance with all collected data.
        """
        hostname = data.get('hostname', 'unknown')

        # Parse memory info
        meminfo = data.get('meminfo', {})
        if meminfo:
            memory = HostMemoryInfo.from_proc_meminfo_dict(meminfo)
        else:
            memory = HostMemoryInfo(total=0, available=None, used=None, free=None,
                                    active=None, inactive=None, buffers=None,
                                    cached=None, shared=None)

        # Parse CPU info
        cpuinfo = data.get('cpuinfo', [])
        cpu = None
        if cpuinfo:
            cpu_summary = summarize_cpuinfo(cpuinfo)
            cpu = HostCPUInfo(
                num_cores=cpu_summary.get('num_physical_cores', 0),
                num_logical_cores=cpu_summary.get('num_logical_cores', 0),
                model=cpu_summary.get('model', ''),
                architecture=cpu_summary.get('architecture', ''),
            )

        # Parse disk info
        diskstats = data.get('diskstats', [])
        disks = None
        if diskstats:
            disks = [HostDiskInfo.from_dict(d) for d in diskstats]

        # Parse network info
        netdev = data.get('netdev', [])
        network = None
        if netdev:
            network = [HostNetworkInfo.from_dict(n) for n in netdev]

        # Parse system info
        loadavg = data.get('loadavg', {})
        system = HostSystemInfo(
            hostname=hostname,
            kernel_version=data.get('version', ''),
            os_release=data.get('os_release', {}),
            uptime_seconds=data.get('uptime_seconds', 0.0),
            load_average_1min=loadavg.get('load_1min', 0.0),
            load_average_5min=loadavg.get('load_5min', 0.0),
            load_average_15min=loadavg.get('load_15min', 0.0),
            running_processes=loadavg.get('running_processes', 0),
            total_processes=loadavg.get('total_processes', 0),
        )

        return cls(
            hostname=hostname,
            memory=memory,
            cpu=cpu,
            disks=disks,
            network=network,
            system=system,
            collection_timestamp=data.get('collection_timestamp'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert HostInfo to a dictionary for JSON serialization."""
        result = {
            'hostname': self.hostname,
            'memory': {
                'total': self.memory.total,
                'available': self.memory.available,
                'used': self.memory.used,
                'free': self.memory.free,
                'active': self.memory.active,
                'inactive': self.memory.inactive,
                'buffers': self.memory.buffers,
                'cached': self.memory.cached,
                'shared': self.memory.shared,
            },
            'collection_timestamp': self.collection_timestamp,
        }

        if self.cpu:
            result['cpu'] = {
                'num_cores': self.cpu.num_cores,
                'num_logical_cores': self.cpu.num_logical_cores,
                'model': self.cpu.model,
                'architecture': self.cpu.architecture,
            }

        if self.disks:
            result['disks'] = [d.to_dict() for d in self.disks]

        if self.network:
            result['network'] = [n.to_dict() for n in self.network]

        if self.system:
            result['system'] = self.system.to_dict()

        return result


class ClusterInformation:
    """
    Comprehensive system information for all hosts in the benchmark environment.

    This class aggregates information from all hosts in a cluster, including
    memory, CPU, disk, and network statistics. It supports multiple collection
    methods: MPI-based collection, DLIO summary extraction, and CLI args.

    Attributes:
        host_info_list: List of HostInfo objects, one per host.
        logger: Logger instance for output.
        total_memory_bytes: Total memory across all hosts in bytes.
        total_cores: Total physical CPU cores across all hosts.
        num_hosts: Number of hosts in the cluster.
        min_memory_bytes: Minimum memory of any single host.
        max_memory_bytes: Maximum memory of any single host.
        collection_method: How data was collected ('mpi', 'dlio_summary', 'args', 'unknown').
        collection_timestamp: ISO timestamp of when data was collected.
        host_consistency_issues: List of detected inconsistencies between hosts.
    """

    def __init__(self, host_info_list: List[HostInfo], logger, calculate_aggregated_info=True):
        self.logger = logger
        self.host_info_list = host_info_list

        # Aggregated information across all hosts
        self.total_memory_bytes = 0
        self.total_cores = 0

        # Extended aggregated attributes
        self.num_hosts = len(host_info_list)
        self.min_memory_bytes = 0
        self.max_memory_bytes = 0
        self.collection_method = "unknown"
        self.collection_timestamp = None
        self.host_consistency_issues: List[str] = []

        if calculate_aggregated_info:
            self.calculate_aggregated_info()

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "total_memory_bytes": self.total_memory_bytes,
            "total_cores": self.total_cores,
            "num_hosts": self.num_hosts,
            "min_memory_bytes": self.min_memory_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "collection_method": self.collection_method,
            "collection_timestamp": self.collection_timestamp,
        }

        # Include host details if available
        if self.host_info_list:
            result["hosts"] = [h.to_dict() if hasattr(h, 'to_dict') else str(h)
                               for h in self.host_info_list]

        # Include consistency issues if any
        if self.host_consistency_issues:
            result["host_consistency_issues"] = self.host_consistency_issues

        return result

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Convert to detailed dictionary including all host information."""
        return self.as_dict()

    @property
    def info(self):
        """Property used by MLPSJsonEncoder for serialization."""
        return self.as_dict()

    def calculate_aggregated_info(self):
        """Calculate aggregated system information across all hosts."""
        if not self.host_info_list:
            return

        memory_values = []
        for host_info in self.host_info_list:
            if hasattr(host_info, 'memory') and host_info.memory:
                self.total_memory_bytes += host_info.memory.total
                memory_values.append(host_info.memory.total)
            if hasattr(host_info, 'cpu') and host_info.cpu:
                self.total_cores += host_info.cpu.num_cores

        if memory_values:
            self.min_memory_bytes = min(memory_values)
            self.max_memory_bytes = max(memory_values)

    def validate_cluster_consistency(self) -> List[str]:
        """
        Check that all nodes have consistent configurations.

        Validates:
        - Memory variance (warns if >10% difference between hosts)
        - CPU core count consistency
        - OS version consistency
        - Kernel version consistency

        Returns:
            List of warning messages for any inconsistencies detected.
        """
        issues = []

        if len(self.host_info_list) < 2:
            return issues

        # Check memory consistency
        memory_values = [h.memory.total for h in self.host_info_list
                        if hasattr(h, 'memory') and h.memory and h.memory.total > 0]
        if memory_values:
            min_mem = min(memory_values)
            max_mem = max(memory_values)
            if min_mem > 0:
                variance = (max_mem - min_mem) / min_mem
                if variance > 0.1:  # >10% difference
                    issues.append(
                        f"Memory variance across hosts: {variance:.1%} "
                        f"(min: {min_mem / (1024**3):.1f}GB, max: {max_mem / (1024**3):.1f}GB)"
                    )

        # Check CPU core count consistency
        core_counts = [h.cpu.num_cores for h in self.host_info_list
                      if hasattr(h, 'cpu') and h.cpu and h.cpu.num_cores > 0]
        if core_counts and len(set(core_counts)) > 1:
            issues.append(
                f"CPU core count varies across hosts: {sorted(set(core_counts))}"
            )

        # Check OS version consistency
        os_versions = []
        for h in self.host_info_list:
            if hasattr(h, 'system') and h.system and h.system.os_release:
                version = h.system.os_release.get('VERSION_ID', '')
                if version:
                    os_versions.append(version)
        if os_versions and len(set(os_versions)) > 1:
            issues.append(
                f"OS version varies across hosts: {sorted(set(os_versions))}"
            )

        # Check kernel version consistency
        kernel_versions = []
        for h in self.host_info_list:
            if hasattr(h, 'system') and h.system and h.system.kernel_version:
                # Extract just the kernel version number
                kv = h.system.kernel_version.split()[2] if len(h.system.kernel_version.split()) > 2 else h.system.kernel_version
                kernel_versions.append(kv)
        if kernel_versions and len(set(kernel_versions)) > 1:
            issues.append(
                f"Kernel version varies across hosts: {sorted(set(kernel_versions))}"
            )

        self.host_consistency_issues = issues
        return issues

    @classmethod
    def from_mpi_collection(cls, collected_data: Dict[str, Any], logger) -> 'ClusterInformation':
        """Create ClusterInformation from MPI collector output.

        Args:
            collected_data: Dictionary from collect_cluster_info(), mapping
                hostname -> system_info dict, plus '_metadata' key.
            logger: Logger instance.

        Returns:
            ClusterInformation instance with full host details.
        """
        host_info_list = []

        # Extract metadata
        metadata = collected_data.pop('_metadata', {})
        collection_method = metadata.get('collection_method', 'mpi')
        collection_timestamp = metadata.get('collection_timestamp')

        # Create HostInfo for each host
        for hostname, host_data in collected_data.items():
            if hostname.startswith('_'):
                continue  # Skip metadata keys
            host_info = HostInfo.from_collected_data(host_data)
            host_info_list.append(host_info)

        # Restore metadata to collected_data (don't mutate caller's data permanently)
        collected_data['_metadata'] = metadata

        inst = cls(host_info_list, logger, calculate_aggregated_info=True)
        inst.collection_method = collection_method
        inst.collection_timestamp = collection_timestamp

        # Validate cluster consistency
        inst.validate_cluster_consistency()

        return inst

    @classmethod
    def from_dlio_summary_json(cls, summary, logger) -> Optional['ClusterInformation']:
        """Create ClusterInformation from DLIO summary.json data.

        Returns None if the required fields are missing from the summary.
        """
        host_memories = summary.get("host_memory_GB")
        host_cpus = summary.get("host_cpu_count")
        if host_memories is None or host_cpus is None:
            return None

        # Create basic HostInfo objects from the summary data
        host_info_list = []
        num_hosts = len(host_memories)
        for i in range(num_hosts):
            memory_bytes = int(host_memories[i] * 1024 * 1024 * 1024)
            host_info = HostInfo(
                hostname=f"host_{i}",
                memory=HostMemoryInfo.from_total_mem_int(memory_bytes),
                cpu=HostCPUInfo(num_cores=host_cpus[i]) if i < len(host_cpus) else None,
            )
            host_info_list.append(host_info)

        inst = cls(host_info_list, logger, calculate_aggregated_info=True)
        inst.collection_method = "dlio_summary"
        return inst

    @classmethod
    def from_dict(cls, data: dict, logger) -> Optional['ClusterInformation']:
        """Create ClusterInformation from a dictionary (e.g., from saved metadata).

        Args:
            data: Dictionary containing cluster information.
            logger: Logger instance.

        Returns:
            ClusterInformation instance, or None if required keys are missing.
        """
        if data is None:
            return None

        total_memory_bytes = data.get("total_memory_bytes")
        if total_memory_bytes is None:
            return None

        # Try to reconstruct host_info_list from saved hosts data
        host_info_list = []
        hosts_data = data.get("hosts", [])
        for host_data in hosts_data:
            if isinstance(host_data, dict):
                host_info = HostInfo.from_collected_data(host_data)
                host_info_list.append(host_info)

        inst = cls(host_info_list, logger, calculate_aggregated_info=False)

        # Restore aggregated values from saved data
        inst.total_memory_bytes = total_memory_bytes
        inst.total_cores = data.get("total_cores", 0)
        inst.num_hosts = data.get("num_hosts", len(host_info_list))
        inst.min_memory_bytes = data.get("min_memory_bytes", 0)
        inst.max_memory_bytes = data.get("max_memory_bytes", 0)
        inst.collection_method = data.get("collection_method", "unknown")
        inst.collection_timestamp = data.get("collection_timestamp")
        inst.host_consistency_issues = data.get("host_consistency_issues", [])

        return inst


class BenchmarkResult:
    """
    Represents the result files from a benchmark run.
    Processes the directory structure to extract metadata and metrics.
    """

    def __init__(self, benchmark_result_root_dir, logger):
        self.benchmark_result_root_dir = benchmark_result_root_dir
        self.logger = logger
        self.metadata = None
        self.summary = None
        self.hydra_configs = {}
        self.issues = []
        self._process_result_directory()

    def _process_result_directory(self):
        """Process the result directory to extract metadata and metrics"""
        # Find and load metadata file
        metadata_files = [f for f in os.listdir(self.benchmark_result_root_dir)
                          if f.endswith('_metadata.json')]

        if metadata_files:
            metadata_path = os.path.join(self.benchmark_result_root_dir, metadata_files[0])
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.verbose(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")

        # Find and load DLIO summary file
        summary_path = os.path.join(self.benchmark_result_root_dir, 'summary.json')
        self.logger.debug(f'Looking for DLIO summary at {summary_path}...')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                self.logger.verbose(f"Loaded DLIO summary from {summary_path}")
            except Exception as e:
                self.logger.error(f"Failed to load DLIO summary from {summary_path}: {e}")

        # Find and load Hydra config files if they exist
        hydra_dir = os.path.join(self.benchmark_result_root_dir, HYDRA_OUTPUT_SUBDIR)
        self.logger.debug(f'Looking for Hydra configs at {hydra_dir}...')
        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            for config_file in os.listdir(hydra_dir):
                if config_file.endswith('.yaml'):
                    config_path = os.path.join(hydra_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            self.hydra_configs[config_file] = yaml.load(f, Loader=yaml.Loader)
                        self.logger.verbose(f"Loaded Hydra config from {config_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Hydra config from {config_path}: {e}")


class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.
    Can be constructed either from a benchmark instance or from result files.
    """
    def __init__(self, benchmark_result=None, benchmark_instance=None, logger=None):
        self.logger = logger
        if benchmark_result is None and benchmark_instance is None:
            self.logger.error(f"The BenchmarkRun instance needs either a benchmark_result or a benchmark_instance.")
            raise ValueError("Either benchmark_result or benchmark_instance must be provided")
        if benchmark_result and benchmark_instance:
            self.logger.error(f"Both benchmark_result and benchmark_instance provided, which is not supported.")
            raise ValueError("Only one of benchmark_result and benchmark_instance can be provided")
            
        self.benchmark_type = None
        self.model = None
        self.accelerator = None
        self.command = None
        self.num_processes = None
        self.parameters = dict()
        self.override_parameters = dict()
        self.system_info = None
        self.metrics = {}
        self._run_id = None
        self._category = None
        self._issues = []
        self.run_datetime = None
        self.result_root_dir = None

        self.benchmark_result = benchmark_result
        self.benchmark_instance = benchmark_instance

        if benchmark_instance:
            self._process_benchmark_instance(benchmark_instance)
            self.post_execution = False
        elif benchmark_result:
            self._process_benchmark_result(benchmark_result)
            self.post_execution = True
        else:
            self.logger.error(f"Neither benchmark_result nor benchmark_instance provided.")
            raise ValueError("Either benchmark_result or benchmark_instance must be provided")

        self._run_id = RunID(program=self.benchmark_type.name, command=self.command,  model=self.model,
                            run_datetime=self.run_datetime)
        self.logger.info(f"Found benchmark run: {self.run_id}")

    @property
    def issues(self):
        return self._issues

    @issues.setter
    def issues(self, issues):
        self._issues = issues

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = category

    @property
    def run_id(self):
        if self.post_execution:
            return self.benchmark_result.benchmark_result_root_dir
        else:
            return self._run_id

    def as_dict(self):
        """Convert the BenchmarkRun object to a dictionary"""
        ret_dict = {
            "run_id": str(self.run_id),
            "benchmark_type": self.benchmark_type.name,
            "model": self.model,
            "command": self.command,
            "num_processes": self.num_processes,
            "parameters": self.parameters,
            "system_info": self.system_info.as_dict() if self.system_info else None,
            "metrics": self.metrics,
        }
        if self.accelerator:
            ret_dict["accelerator"] = str(self.accelerator)

        return ret_dict

    def _process_benchmark_instance(self, benchmark_instance):
        """Extract parameters and system info from a running benchmark instance"""
        self.benchmark_type = benchmark_instance.BENCHMARK_TYPE
        self.model = getattr(benchmark_instance.args, 'model', None)
        self.command = getattr(benchmark_instance.args, 'command', None)
        self.run_datetime = benchmark_instance.run_datetime
        self.num_processes = benchmark_instance.args.num_processes
        
        # Extract parameters from the benchmark instance
        if hasattr(benchmark_instance, 'combined_params'):
            self.parameters = benchmark_instance.combined_params
        else:
            # Fallback to args if combined_params not available
            self.parameters = vars(benchmark_instance.args)

        self.override_parameters = benchmark_instance.params_dict
            
        # Extract system information
        if hasattr(benchmark_instance, 'cluster_information'):
            self.system_info = benchmark_instance.cluster_information

    def _process_benchmark_result(self, benchmark_result):
        """Extract parameters and system info from result files"""
        # Process the summary and hydra configs to find what was run
        summary_workload = benchmark_result.summary.get('workload', {})
        hydra_workload_config = benchmark_result.hydra_configs.get("config.yaml", {}).get("workload", {})
        hydra_workload_overrides = benchmark_result.hydra_configs.get("overrides.yaml", {})
        hydra_workflow = hydra_workload_config.get("workflow", {})
        workflow = (
            hydra_workflow.get('generate_data', {}),
            hydra_workflow.get('train', {}),
            hydra_workflow.get('checkpoint', {}),
        )
        workloads = [i for i in hydra_workload_overrides if i.startswith('workload=')]

        # Get benchmark type based on workflow
        if workflow[0] or workflow[1]:
            # Unet3d can have workflow[2] == True but it'll get caught here first
            self.benchmark_type = BENCHMARK_TYPES.training
        elif workflow[2]:
            self.benchmark_type = BENCHMARK_TYPES.checkpointing

        # The model for checkpointing in dlio doesn't have the "3" and we match against inputs to the cli which
        # use a hypen instead of an underscore. We should make this better in the next version
        # TODO: Make this better
        self.model = hydra_workload_config.get('model', {}).get("name")
        self.model = self.model.replace("llama_", "llama3_")
        self.model = self.model.replace("_", "-")

        self.num_processes = benchmark_result.summary["num_accelerators"]

        # Set command for training
        if self.benchmark_type == BENCHMARK_TYPES.training:
            if workflow[1]:
                # If "workflow.train" is present, even if there is checkpoint or datagen, it's a run_benchmark.
                # When running DLIO with datagen and run in a single run, the metrics are still available separately
                self.command = "run_benchmark"
                self.accelerator = workloads[0].split('_')[1]
            if workflow[0]:
                # If we don't get caught by run and workflow[0] (datagen) is True, then we have a datagen command
                self.command = "datagen"

        self.run_datetime = benchmark_result.summary.get("start")
        self.parameters = benchmark_result.hydra_configs.get("config.yaml", {}).get("workload", {})

        for param in benchmark_result.hydra_configs.get("overrides.yaml", list()):
            p, v = param.split('=')
            if p.startswith('++workload.'):
                self.override_parameters[p[len('++workload.'):]] = v

        self.metrics = benchmark_result.summary.get("metric")

        # Try to get system_info from DLIO summary first, then fall back to metadata
        self.system_info = ClusterInformation.from_dlio_summary_json(benchmark_result.summary, self.logger)
        if self.system_info is None and benchmark_result.metadata:
            # Fall back to cluster_information from metadata if DLIO summary lacks it
            cluster_info_dict = benchmark_result.metadata.get("cluster_information")
            self.system_info = ClusterInformation.from_dict(cluster_info_dict, self.logger)


class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify call the self.check_* methods
    """
    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        self.issues = []
        
        # Dynamically find all check methods
        self.check_methods = [getattr(self, method) for method in dir(self) 
                             if callable(getattr(self, method)) and method.startswith('check_')]
        
    def run_checks(self) -> List[Issue]:
        """Run all check methods and return a list of issues"""
        self.issues = []
        for check_method in self.check_methods:
            try:
                self.logger.debug(f"Running check {check_method.__name__}")
                method_issues = check_method()
                if method_issues:
                    if isinstance(method_issues, list):
                        self.issues.extend(method_issues)
                    else:
                        self.issues.append(method_issues)
            except Exception as e:
                self.logger.error(f"Error running check {check_method.__name__}: {e}")
                self.issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Check {check_method.__name__} failed with error: {e}",
                    severity="error"
                ))
        
        return self.issues


class RunRulesChecker(RulesChecker):
    """
    This class verifies rules against individual benchmark runs.
    """

    def __init__(self, benchmark_run, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_run = benchmark_run


class MultiRunRulesChecker(RulesChecker):
    """Rules checker for multiple benchmark runs as for a single workload or for the full submission"""

    def __init__(self, benchmark_runs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(benchmark_runs) not in [list, tuple]:
            raise TypeError("benchmark_runs must be a list or tuple")
        self.benchmark_runs = benchmark_runs

    def check_runs_valid(self) -> Optional[Issue]:
        category_set = {run.category for run in self.benchmark_runs}
        if PARAM_VALIDATION.INVALID in category_set:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Invalid runs found.",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif PARAM_VALIDATION.OPEN in category_set:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="All runs satisfy the OPEN or CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif {PARAM_VALIDATION.CLOSED} == category_set:
            return Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message="All runs satisfy the CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        return None


class TrainingRunRulesChecker(RunRulesChecker):
    """Rules checker for training benchmarks"""
    
    def check_benchmark_type(self) -> Optional[Issue]:
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.training:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.training,
                actual=self.benchmark_run.benchmark_type
            )
        return None
    
    def check_num_files_train(self) -> Optional[Issue]:
        """Check if the number of training files meets the minimum requirement"""
        if 'dataset' not in self.benchmark_run.parameters:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing dataset parameters",
                parameter="dataset"
            )
            
        dataset_params = self.benchmark_run.parameters['dataset']
        if 'num_files_train' not in dataset_params:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing num_files_train parameter",
                parameter="dataset.num_files_train"
            )
            
        # Calculate required file count based on system info
        # This is a simplified version - in practice you'd use the calculate_training_data_size function
        configured_num_files = int(dataset_params['num_files_train'])
        dataset_params = self.benchmark_run.parameters['dataset']
        reader_params = self.benchmark_run.parameters['reader']
        required_num_files, _, _ = calculate_training_data_size(None, self.benchmark_run.system_info, dataset_params, reader_params, self.logger, self.benchmark_run.num_processes)

        if configured_num_files < required_num_files:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient number of training files",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        
        return None

    def check_allowed_params(self) -> Optional[Issue]:
        """
        This method will verify that the only parameters that were set were the allowed parameters.
        Allowed for closed:
          - dataset.num_files_train
          - dataset.num_subfolders_train
          -
        :return:
        """
        closed_allowed_params = ['dataset.num_files_train', 'dataset.num_subfolders_train', 'dataset.data_folder',
                                 'reader.read_threads', 'reader.computation_threads', 'reader.transfer_size',
                                 'reader.odirect', 'reader.prefetch_size', 'checkpoint.checkpoint_folder',
                                 'storage.storage_type', 'storage.storage_root']
        open_allowed_params = ['framework', 'dataset.format', 'dataset.num_samples_per_file', 'reader.data_loader']
        issues = []
        for param, value in self.benchmark_run.override_parameters.items():
            if param.startswith("workflow"):
                # We handle workflow parameters separately
                continue
            self.logger.debug(f"Processing override parameter: {param} = {value}")
            if param in closed_allowed_params:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Closed parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            elif param in open_allowed_params:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Open parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Disallowed parameter override: {param} = {value}",
                    parameter="Overrode Parameters",
                    expected="None",
                    actual=value
                ))
        return issues

    def check_workflow_parameters(self) -> Optional[Issue]:
        issues = []
        # Check if the workflow parameters are valid
        workflow_params = self.benchmark_run.parameters.get('workflow', {})
        for param, value in workflow_params.items():
            if self.benchmark_run.model == UNET and self.benchmark_run.command == "run_benchmark":
                # Unet3d training requires the checkpoint workflow = True
                if param == "checkpoint":
                    if value == True:
                        return Issue(
                            validation=PARAM_VALIDATION.CLOSED ,
                            message="Unet3D training requires executing a checkpoing",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )
                    elif value == False:
                        return Issue(
                            validation=PARAM_VALIDATION.INVALID,
                            message="Unet3D training requires executing a checkpoint. The parameter 'workflow.checkpoint' is set to False",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )
        return None

    def check_odirect_supported_model(self) -> Optional[Issue]:
        # The 'reader.odirect' option is only supported if the model is "Unet3d"
        if self.benchmark_run.model != UNET and self.benchmark_run.parameters.get('reader', {}).get('odirect'):
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="The reader.odirect option is only supported for Unet3d model",
                parameter="reader.odirect",
                expected="False",
                actual=self.benchmark_run.parameters.get('reader', {}).get('odirect')
            )
        else:
            return None

    def check_checkpoint_files_in_code(self) -> Optional[Issue]:
        pass

    def check_num_epochs(self) -> Optional[Issue]:
        pass

    def check_inter_test_times(self) -> Optional[Issue]:
        pass

    def check_file_system_caching(self) -> Optional[Issue]:
        pass


class CheckpointingRunRulesChecker(RunRulesChecker):
    """Rules checker for checkpointing benchmarks"""
    def check_benchmark_type(self) -> Optional[Issue]:
        pass


#######################################################################################################################
# Define the checkers for groups of runs representing submissions
#######################################################################################################################

class CheckpointSubmissionRulesChecker(MultiRunRulesChecker):
    supported_models = LLM_MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 10 total writes and 10 total reads for checkpointing benchmarks.  It's possible for a submitter
         to have a single run with all checkpoints, two runs that separate reads and writes, or individual runs
         for each read and write operation.
        """
        issues = []
        num_writes = num_reads = 0
        for run in self.benchmark_runs:
            if run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                num_writes += run.parameters.get('checkpoint', {}).get('num_checkpoints_write', 0)
                num_reads += run.parameters.get('checkpoint', {}).get('num_checkpoints_read', 0)

        if not num_reads == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected 10 total read operations, but found {num_reads}",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=num_reads
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total read operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=num_reads
            ))

        if not num_writes == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected 10 total write operations, but found {num_writes}",
                parameter="checkpoint.num_checkpoints_write",
                expected=10,
                actual=num_writes
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total write operations",
                parameter="checkpoint.num_checkpoints_write",
                expected=10,
                actual=num_writes
            ))

        if num_writes == 10 and num_reads == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total read and write operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=10,
            ))

        return issues


class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    supported_models = MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 5 runs for training benchmarks
        """

class BenchmarkVerifier:

    def __init__(self, *benchmark_runs, logger=None):
        self.logger = logger
        self.issues = []

        if len(benchmark_runs) == 1:
            self.mode = "single"
        elif len(benchmark_runs) > 1:
            self.mode = "multi"
        else:
            raise ValueError("At least one benchmark run is required")

        self.benchmark_runs = benchmark_runs
        if self.mode == "single":
            if "mlpstorage.benchmarks." in str(type(benchmark_runs[0])):
                # This is here if we get a Benchmark instance that needs to run the verifier
                # on itself before execution. We map it to a BenchmarkRun instance
                # We check against the string so we don't need to import the Benchmark classes here
                self.benchmark_runs = [BenchmarkRun(benchmark_instance=benchmark_runs[0], logger=logger)]

            benchmark_run = self.benchmark_runs[0]
            if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger)
            elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger)

        elif self.mode == "multi":
            benchmark_types = {br.benchmark_type for br in benchmark_runs}
            if len(benchmark_types) > 1:
                raise ValueError("Multi-run verification requires all runs are from the same benchmark type. Got types: {benchmark_types}")
            else:
                benchmark_type = benchmark_types.pop()

            if benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingSubmissionRulesChecker(benchmark_runs, logger)
            if benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointSubmissionRulesChecker(benchmark_runs, logger)

    def verify(self) -> PARAM_VALIDATION:
        run_ids = [br.run_id for br in self.benchmark_runs]
        if self.mode == "single":
            self.logger.status(f"Verifying benchmark run for {run_ids[0]}")
        elif self.mode == "multi":
            self.logger.status(f"Verifying benchmark runs for {', '.join(run_ids)}")
        self.issues = self.rules_checker.run_checks()
        num_invalid = 0
        num_open = 0
        num_closed = 0

        for issue in self.issues:
            if issue.validation == PARAM_VALIDATION.INVALID:
                self.logger.error(f"INVALID: {issue}")
                num_invalid += 1
            elif issue.validation == PARAM_VALIDATION.CLOSED:
                self.logger.status(f"Closed: {issue}")
                num_closed += 1
            elif issue.validation == PARAM_VALIDATION.OPEN:
                self.logger.status(f"Open: {issue}")
                num_open += 1
            else:
                raise ValueError(f"Unknown validation type: {issue.validation}")

        if self.mode == "single":
            self.benchmark_runs[0].issues = self.issues

        if num_invalid > 0:
            self.logger.status(f'Benchmark run is INVALID due to {num_invalid} issues ({run_ids})')
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.INVALID
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.OPEN
            self.logger.status(f'Benchmark run qualifies for OPEN category ({run_ids})')
            return PARAM_VALIDATION.OPEN
        else:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.CLOSED
            self.logger.status(f'Benchmark run qualifies for CLOSED category ({run_ids})')
            return PARAM_VALIDATION.CLOSED


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger,
                                 num_processes=None) -> Tuple[int, int, int]:
    """
    Validate the parameters for the datasize operation and apply rules for a closed submission.

    Requirements:
      - Dataset needs to be 5x the amount of total memory
      - Training needs to do at least 500 steps per epoch

    Memory Ratio:
      - Collect "Total Memory" from /proc/meminfo on each host
      - sum it up
      - multiply by 5
      - divide by sample size
      - divide by batch size

    500 steps:
      - 500 steps per ecpoch
      - multiply by max number of processes
      - multiply by batch size

    If the number of files is greater than MAX_NUM_FILES_TRAIN, use the num_subfolders_train parameters to shard the
    dataset.
    :return:
    """
    required_file_count = 1
    required_subfolders_count = 0

    # Find the amount of memory in the cluster via args or measurements
    if not args:
        total_mem_bytes = cluster_information.total_memory_bytes
    elif args.client_host_memory_in_gb and args.num_client_hosts:
        # If host memory per client and num clients is provided, we use these values instead of the calculated memory
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    elif args.clienthost_host_memory_in_gb and not args.num_client_hosts:
        # If we have memory but not clients, we use the number of provided hosts and given memory amount
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    else:
        raise ValueError('Either args or cluster_information is required')

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes
    file_size_bytes = dataset_params['num_samples_per_file'] * dataset_params['record_length_bytes']

    min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024} MB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')

    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


"""
The results directory structure is as follows:
results_dir:
    <benchmark_name>:
        <model>:
            <command>:
                    <datetime>:
                        run_<run_number> (Optional)
                    
This looks like:
results_dir:
    training:
        unet3d:
            datagen:
                <datetime>:
                    <output_files>
            run:
                <datetime>:
                    <output_files>
    checkpointing:
        llama3-8b:
            <datetime>:
                <output_files>
"""



def generate_output_location(benchmark, datetime_str=None, **kwargs):
    """
    Generate a standardized output location for benchmark results.

    Output structure follows this pattern:
    RESULTS_DIR:
        <benchmark_name>:
            <model>:
                <command>:
                        <datetime>:
                            run_<run_number> (Optional)

    Args:
        benchmark (Benchmark): benchmark (e.g., 'training', 'vectordb', 'checkpoint')
        datetime_str (str, optional): Datetime string for the run. If None, current datetime is used.
        **kwargs: Additional benchmark-specific parameters:
            - model (str): For training benchmarks, the model name (e.g., 'unet3d', 'resnet50')
            - category (str): For vectordb benchmarks, the category (e.g., 'throughput', 'latency')

    Returns:
        str: The full path to the output location
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    output_location = benchmark.args.results_dir
    if hasattr(benchmark, "run_number"):
        run_number = benchmark.run_number
    else:
        run_number = 0

    # Handle different benchmark types
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, datetime_str)

    else:
        print(f'The given benchmark is not supported by mlpstorage.rules.generate_output_location()')
        sys.exit(1)

    return output_location


def get_runs_files(results_dir, logger=None):
    """
    Walk the results_dir location and return a list of BenchmarkResult objects that represent a single run

    :param results_dir: Base directory containing benchmark results
    :param benchmark_name: Optional filter for specific benchmark name
    :param command: Optional filter for specific command
    :return: List of dictionaries with run information
    """
    if logger is None:
        logger = setup_logging(name='mlpstorage.rules.get_runs_files')

    if not os.path.exists(results_dir):
        logger.warning(f'Results directory {results_dir} does not exist.')
        return []

    runs = []

    # Walk through all directories and files in results_dir
    for root, dirs, files in os.walk(results_dir):
        logger.ridiculous(f'Processing directory: {root}')

        # Look for metadata files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]

        if not metadata_files:
            logger.debug(f'No metadata file found')
            continue
        else:
            logger.debug(f'Found metadata files in directory {root}: {metadata_files}')

        if len(metadata_files) > 1:
            logger.warning(f'Multiple metadata files found in directory {root}. Skipping this directory.')
            continue

        # Find DLIO summary.json file if it exists
        dlio_summary_file = None
        for f in files:
            if f == 'summary.json':
                dlio_summary_file = os.path.join(root, f)
                break

        if dlio_summary_file:
            runs.append(BenchmarkRun(benchmark_result=BenchmarkResult(root, logger), logger=logger))

    return runs

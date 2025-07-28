import abc
import enum
import json
import os
import pdb
import yaml

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pprint import pprint, pformat
from statistics import mean
from typing import List, Dict, Any, Optional, Tuple, Union

from mlpstorage.config import (MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, BENCHMARK_TYPES,
                               DATETIME_STR, LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, HYDRA_OUTPUT_SUBDIR, UNET, RESNET,
                               COSMOFLOW, AU_REQUIREMENT, LLM_SIZE_BY_RANK, CLOSED, OPEN)
from mlpstorage.mlps_logging import setup_logging
from mlpstorage.utils import is_valid_datetime_format


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
        """Create a HostMemoryInfo instance from a dictionary"""
        converted_dict = dict(
            total=data.get('MemTotal', 0) * 1024,
            available=data.get('MemAvailable', 0) * 1024,
            used=data.get('MemUsed', 0) * 1024,
            free=data.get('MemFree', 0) * 1024,
            active=data.get('Active', 0) * 1024,
            inactive=data.get('Inactive', 0) * 1024,
            buffers=data.get('Buffers', 0) * 1024,
            cached=data.get('Cached', 0) * 1024,
            shared=data.get('Shmem', 0) * 1024
        )
        converted_dict = {k: int(v.split(" ")[0]) for k, v in converted_dict.items()}
        return cls(**converted_dict)

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
    """Information about a single host in the system"""
    hostname: str
    memory: HostMemoryInfo = field(default_factory=HostMemoryInfo)
    cpu: Optional[HostCPUInfo] = None

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


class ClusterInformation:
    """
    Comprehensive system information for all hosts in the benchmark environment.
    This includes detailed memory, CPU, and accelerator information.
    """

    def __init__(self, host_info_list: List[str], logger, calculate_aggregated_info=True):
        self.logger = logger
        self.host_info_list = host_info_list

        # Aggregated information across all hosts
        self.total_memory_bytes = 0
        self.total_cores = 0

        if calculate_aggregated_info:
            self.calculate_aggregated_info()

    def as_dict(self):
        return {
            "total_memory_bytes": self.total_memory_bytes,
            "total_cores": self.total_cores,
        }

    def calculate_aggregated_info(self):
        """Calculate aggregated system information across all hosts"""
        for host_info in self.host_info_list:
            self.total_memory_bytes += host_info.memory.total
            if host_info.cpu:
                self.total_cores += host_info.cpu.num_cores

    @classmethod
    def from_dlio_summary_json(cls, summary, logger) -> 'ClusterInformation':
        host_memories = summary.get("host_memory_GB")
        host_cpus = summary.get("host_cpu_count")
        num_hosts = summary.get("num_hosts")
        host_info_list = []
        inst = cls(host_info_list, logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = sum(host_memories) * 1024 * 1024 * 1024
        inst.total_cores = sum(host_cpus)
        return inst



class BenchmarkResult:
    """
    Represents the result files from a benchmark run.
    Processes the directory structure to extract metadata and metrics.
    """

    def __init__(self, benchmark_result_root_dir, logger):
        self.benchmark_result_root_dir = benchmark_result_root_dir
        self.submitter_metadata = {}

        self.logger = logger
        self.metadata = None
        self.summary = None
        self.system_description = None
        self.system_description_error = None
        self.expected_system_description_yaml_path = None
        self.hydra_configs = {}
        self.per_rank_per_epoch_stats = {}
        self.per_rank_outputs = {}
        self.issues = []

        if os.path.exists("submitter_path_indexes.json"):
            with open("submitter_path_indexes.json", "r") as f:
                self.SUBMITTER_SYSTEM_PATH_INDEXES = json.load(f)
        else:
            self.SUBMITTER_SYSTEM_PATH_INDEXES = None

        self._process_result_directory()


    def _process_result_directory(self):
        """Process the result directory to extract metadata and metrics"""
        self._load_metadata_file()
        self._load_dlio_summary_file()
        self._load_hydra_configs()
        self._load_per_rank_per_epoch_stats()
        # self._load_per_rank_outputs()
        if self.SUBMITTER_SYSTEM_PATH_INDEXES:
            self._extract_submitter_metadata()
        self._load_system_yaml()

    def _extract_submitter_metadata(self):
        try:
            split_path = self.benchmark_result_root_dir.split('/')
            if split_path[1] in self.SUBMITTER_SYSTEM_PATH_INDEXES.keys():
                indexes = self.SUBMITTER_SYSTEM_PATH_INDEXES[split_path[1]]
            else:
                indexes = self.SUBMITTER_SYSTEM_PATH_INDEXES['default']

            for data, index  in indexes.items():
                if not index:
                    continue

                self.submitter_metadata[data] = split_path[index]
        except Exception as e:
            import pdb
            pdb.set_trace()

    def _load_metadata_file(self):
        # Find and load metadata file
        metadata_files = [f for f in os.listdir(self.benchmark_result_root_dir)
                          if f.endswith('_metadata.json')]

        if metadata_files:
            metadata_path = os.path.join(self.benchmark_result_root_dir, metadata_files[0])
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.ridiculous(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")

    def _load_system_yaml(self):
        # This is going to be annoying because one submitter has what I'm calling "system_group" and they handle
        #  it in a way that I don't like. The system yaml file is named for the group, not the system
        import pdb
        try:
            split_path = self.benchmark_result_root_dir.split('/')
            # For a given run, we want to go to the results level, change to systems, and read the correct systems.yaml file
            for result_dir_name in ["results", "result"]:
                if result_dir_name in split_path:
                    result_index = split_path.index(result_dir_name)
                    break
            else:
                pdb.set_trace()

            if self.submitter_metadata["system_name"] == "weka-hpe-12-client-converged-resnet-h100":
                import pdb
                pdb.set_trace()

            system_path_list = split_path[:result_index]
            system_path = os.path.join(*system_path_list)

            # Now we find the system path. It could be "system" or "systems"
            for system_dir_name in ["system", "systems"]:
                if os.path.isdir(os.path.join(system_path, system_dir_name)):
                    system_path = os.path.join(system_path, system_dir_name)
                    break

            systems_names = [f[:-len(".yaml")] for f in os.listdir(system_path) if f.endswith('.yaml')]
            json_system_names = [f[:-len(".json")] for f in os.listdir(system_path) if f.endswith('.json')]

            # Now we try to match the system_name to the systems.yaml files we found
            # One submitter has "system_group" and this is what we're doing for that
            if system_group := self.submitter_metadata.get('system_group'):
                check_filename = f"{system_group}.yaml"
            else:
                check_filename = f"{self.submitter_metadata['system_name']}.yaml"


            # Check for the exact match first:
            if os.path.isfile(os.path.join(system_path, check_filename)):
                system_yaml_path = os.path.join(system_path, check_filename)
                self.logger.verbose(f'Found system YAML at {system_yaml_path}')
                self._read_system_description_yaml(system_yaml_path)
                return
            elif os.path.isfile(os.path.join(system_path, check_filename.lower())):
                system_yaml_path = os.path.join(system_path, check_filename.lower())
                self.logger.verbose(f'Found system YAML at {system_yaml_path}')
                self._read_system_description_yaml(system_yaml_path)
                return

            # Check for exact match with a json file, if it exists
            check_file_path_json = os.path.join(system_path, f"{self.submitter_metadata["system_name"]}.json")
            if os.path.isfile(check_file_path_json):
                system_yaml_path = check_file_path_json
                self.logger.verbose(f'Found system YAML at {system_yaml_path}')
                self._read_system_description_yaml(system_yaml_path)
                return

            # If no exact match, try to match the system_name to the systems.yaml files we found
            # Check if self.submitter_metadata['system_name'] is a substring of any of the systems.yaml files we found
            # If the system_name is a substring of any of the systems.yaml files, use that
            for system_name in systems_names:
                self.logger.ludicrous(f'Checking {system_name} in {self.submitter_metadata["system_name"]}...')
                if (system_name.lower() in self.submitter_metadata['system_name'].lower()
                    or self.submitter_metadata['system_name'].lower() in system_name.lower()):
                    system_yaml_path = os.path.join(system_path, f"{system_name}.yaml")
                    self.logger.verbose(f'Found system YAML at {system_yaml_path}')
                    self._read_system_description_yaml(system_yaml_path)
                    return

            self.logger.error(f'No system YAML found for {self.submitter_metadata["system_name"]} in {systems_names}')
            self.expected_system_description_yaml_path = os.path.join(system_path, check_filename)
            self.system_description_error = f"No system YAML found."

        except Exception as e:
            pdb.post_mortem()

    def _read_system_description_yaml(self, system_yaml_path):
        with open(system_yaml_path, 'r') as f:
            try:
                self.expected_system_description_yaml_path = system_yaml_path
                if system_yaml_path.endswith('.json'):
                    self.logger.warning(f'Found system YAML at {system_yaml_path} in JSON format')
                    self.system_description = json.load(f)
                else:
                    self.logger.info(f'Found system YAML at {system_yaml_path} in YAML format')
                    self.system_description = yaml.safe_load(f)
            except yaml.scanner.ScannerError as e:
                self.logger.error(f"Improperly formatted yaml: {system_yaml_path} - {e}")
                self.system_description_error = f"Improperly formatted yaml: {system_yaml_path} - {e}"
            except Exception as e:
                import pdb
                pdb.post_mortem()
                self.system_description = str(e)

    def _load_dlio_summary_file(self):
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

    def _load_hydra_configs(self):
        # Find and load Hydra config files if they exist
        hydra_dir = os.path.join(self.benchmark_result_root_dir, HYDRA_OUTPUT_SUBDIR)
        self.logger.debug(f'Looking for Hydra configs at {hydra_dir}...')
        # This should find config.yaml, hydra.yaml, and overrides.yaml as of DLIO for MLPS v2.0
        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            for config_file in os.listdir(hydra_dir):
                if config_file.endswith('.yaml'):
                    config_path = os.path.join(hydra_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            self.hydra_configs[config_file] = yaml.load(f, Loader=yaml.Loader)
                        self.logger.ridiculous(f"Loaded Hydra config from {config_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Hydra config from {config_path}: {e}")

    def _load_per_rank_per_epoch_stats(self):
        # Find rank_per_epoch_stats json documents
        # Data in EACH file will be in the format:
        #  {
        #       "<epoch_num>": {
        #           "start": "<start_time>",
        #           "end": "<end_time>",
        #           "duration": <time_in_seconds>,
        #
        #           # For Training workloads we see block section with multiple epochs.
        #           "block1": {
        #               "start": "<start_time>",
        #               "end": "<end_time>",
        #               "duration": <time_in_seconds>
        #           },
        #
        #
        #           # For Checkpointing workloads we see save_ and load_ sections with a single epoch:
        #           "save_ckpt1": {
        #               "start": "<start_time>",
        #               "end": "<end_time>",
        #               "duration": <time_in_seconds>,
        #               "throughput": <throughput_in_GBps>,
        #           },
        #           "load_ckpt1": {
        #               "start": "<start_time>",
        #               "end": "<end_time>",
        #               "duration": <time_in_seconds>.
        #               "throughput": <throughput_in_GBps>,
        #           }
        #       }
        #  }

        # This should find rank_per_epoch_stats.json files as of DLIO for MLPS v2.0
        rank_per_epoch_stats_files = [f for f in os.listdir(self.benchmark_result_root_dir)
                                      if f.endswith('_per_epoch_stats.json')]
        self.per_rank_per_epoch_stats = self._read_per_rank_json_files(rank_per_epoch_stats_files)

    def _load_per_rank_outputs(self):
        # This should find rank_per_output.json files as of DLIO for MLPS v2.0
        rank_output_files = [f for f in os.listdir(self.benchmark_result_root_dir)
                                      if f.endswith('_output.json')]
        self.per_rank_outputs = self._read_per_rank_json_files(rank_output_files)

    def _read_per_rank_json_files(self, per_rank_files):
        per_rank_content = {}
        for rank_file in per_rank_files:
            rank = rank_file.split('_')[0]
            rank_output_path = os.path.join(self.benchmark_result_root_dir, rank_file)
            try:
                with open(rank_output_path, 'r') as f:
                    rank_output = json.load(f)
                per_rank_content[rank] = rank_output
                self.logger.ridiculous(f"Loaded rank_output from {rank_output_path}")
            except Exception as e:
                self.logger.error(f"Failed to load rank_output from {rank_output_path}: {e}")

        return per_rank_content


class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.
    Can be constructed either from a benchmark instance or from result files.

    The purpose of this class is to provide a unified interface for RulesCheckers to run before a benchmark runs
    and on the result files of a previously run benchmark. The interface to mlpstorage doesn't completely match
    the output format of DLIO.

    I'm not a big fan of this double abstraction and it should get refactored in the next release. The problem was
    that this code was developed with the understanding that a benchmark could only be run with mlpstorage. Late in
    the process I realized that DLIO could be run directly and we need to be able to read the results from DLIO
    without the benefit of the metadata that mlpstorage writes. This was the least work method to support the
    manual-DLIO use case, it was not the best design.
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
        self.num_hosts = None
        self.parameters = dict()
        self.override_parameters = dict()
        self.system_info = None
        self.system_description = dict()
        self.system_description_error = ""
        self.expected_system_description_yaml_path = None
        self.metrics = {}
        self._run_id = None
        self._category = None
        self._issues = []
        self.run_datetime = None
        self.result_root_dir = None

        self.submitter = None
        self.submitted_category = None
        self.system_name = None
        self.device_name = None

        self.benchmark_result = benchmark_result
        self.benchmark_instance = benchmark_instance

        if benchmark_instance:
            self._process_benchmark_instance(benchmark_instance)
            self.post_execution = False
        elif benchmark_result:
            self._process_benchmark_result(benchmark_result)
            self._process_result_metrics()
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
            "submitter": self.submitter,
            "system_name": self.system_name,
            "submitted_category": self.submitted_category,
            "benchmark_type": self.benchmark_type.name,
            "model": self.model,
            "command": self.command,
            "num_processes": self.num_processes,
            "parameters": self.parameters,
            "system_info": self.system_info.as_dict() if self.system_info else None,
            "metrics": self.metrics,
        }

        add_err = ""
        if self.benchmark_result.system_description_error:
            add_err = self.benchmark_result.system_description_error
            add_err = add_err.replace("\n", " ")
            add_err = add_err.replace(",", " | ")
            self.logger.verboser(f"Error parsing system description: {add_err}")
        elif isinstance(self.system_description, str):
            # This means we got an error and wan tto keep it as part of the error
            ret_dict["system_description"] = self.system_description
            add_err = self.system_description
            self.system_description = ""
        elif isinstance(self.system_description, dict):
            # This means we have what we expect and don't need to handle an existing error.
            for system_description_key in ["System", "system", "storage_system"]:
                if system_description_key in self.system_description.keys():
                    ret_dict["System"] = self.system_description[system_description_key]
                    break
            else:
                add_err = f"Error parsing system info. Keys: {', '.join(self.system_description.keys())}"
        elif isinstance(self.system_description, Exception):
            self.logger.verboser(f"Error parsing system info. Expected Str or Dict. Got {type(self.system_description)}")
            add_err = str(self.system_description)
        elif self.system_description is None:
            self.system_description = ""
            add_err = f"Unable to parse system description file."
        else:
            add_err = f"Error parsing system info. Expected Str or Dict. Got {type(self.system_description)}"

        if self.system_description_error:
            self.logger.verboser(f"Error parsing system description: {self.system_description_error}")
            self.system_description_error += add_err
        else:
            self.system_description_error = add_err
        ret_dict["system_description_error"] = add_err

        for k, v in self.__dict__.items():
            if type(v) in [str, int, float, bool]:
                ret_dict[k] = v

        if self.benchmark_result:
            ret_dict['summary'] = self.benchmark_result.summary

        if self.accelerator:
            ret_dict["accelerator"] = str(self.accelerator)

        if "System" not in ret_dict.keys():
            ret_dict["System"] = "No data found"
        return ret_dict

    def _process_benchmark_instance(self, benchmark_instance):
        """Extract parameters and system info from a running benchmark instance"""
        self.benchmark_type = benchmark_instance.BENCHMARK_TYPE
        self.model = getattr(benchmark_instance.args, 'model', None)
        self.command = getattr(benchmark_instance.args, 'command', None)
        self.run_datetime = benchmark_instance.run_datetime
        self.num_processes = benchmark_instance.args.num_processes
        self.num_hosts = len(benchmark_instance.args.hosts)
        
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
        if benchmark_result.submitter_metadata:
            self.submitter = benchmark_result.submitter_metadata.get('submitter', None)
            self.submitted_category = benchmark_result.submitter_metadata.get('submitted_category', CLOSED).lower()
            self.system_name = benchmark_result.submitter_metadata.get('system_name', None)
            self.system_group = benchmark_result.submitter_metadata.get('system_group', None)

            if self.system_group:
                self.device_name = self.system_name
                self.system_name = f"{self.system_name} - {self.system_group}"

            self.full_system_name = f"{self.submitter} - {self.system_name}"

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
        self.num_hosts = benchmark_result.summary.get("num_hosts", 1)

        # Set command for training
        if self.benchmark_type == BENCHMARK_TYPES.training:
            if workflow[1]:
                # If "workflow.train" is present, even if there is checkpoint or datagen, it's a run_benchmark.
                # When running DLIO with datagen and run in a single run, the metrics are still available separately
                #   This is only if DLIO is run manually, mlpstorage can't do train & datagen in a single command
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

        self.system_info = ClusterInformation.from_dlio_summary_json(benchmark_result.summary, self.logger)
        self.system_description = benchmark_result.system_description
        self.expected_system_description_yaml_path = benchmark_result.expected_system_description_yaml_path

    def _process_result_metrics(self):
        if self.benchmark_type == BENCHMARK_TYPES.training:
            # Extract metrics from the benchmark result
            summary_metrics = self.benchmark_result.summary.get('metric', {})
            training_metrics = dict(
                train_au_mean_percentage=summary_metrics.get('train_au_mean_percentage'),
                train_au_meet_expectation=summary_metrics.get('train_au_meet_expectation'),
                train_throughput_samples_per_second=summary_metrics.get('train_throughput_samples_per_second'),
                train_io_mean_MB_per_second=summary_metrics.get('train_io_mean_MB_per_second'),
                num_accelerators=self.benchmark_result.summary.get('num_accelerators'),
            )
            self.metrics = training_metrics

        if self.benchmark_type == BENCHMARK_TYPES.checkpointing:
            # Here we need to pull the minimum throughput for each checkpoint across all ranks
            # This will be nested dicts of {<checkpoint_operation>: {<checkpoint_number>: List[Dict]}}
            checkpoint_dicts = dict(save=dict(), load=dict())
            max_durations = dict(save=dict(), load=dict())
            calculated_max_duration = dict(save=dict(), load=dict())
            datetime_str_format = "%Y-%m-%dT%H:%M:%S.%f"
            start_times = dict(save=dict(), load=dict())
            end_times = dict(save=dict(), load=dict())
            durations = dict(save=dict(), load=dict())
            throughputs = dict(save=dict(), load=dict())
            sizes = dict(save=dict(), load=dict())
            for rank, stats in self.benchmark_result.per_rank_per_epoch_stats.items():
                # We have epoch set to 1 by default
                rank_dicts = stats.get("1")
                for key, value in rank_dicts.items():
                    if key.startswith("save_") or key.startswith('load_'):
                        # We have another nested dictionary for saving or loading
                        checkpoint_operation = key.split('_')[0]
                        checkpoint_number = int(key[len("save_ckpt"):])
                        dict_to_append = {
                            "rank": int(rank),
                            "checkpoint_operation": checkpoint_operation,
                            "checkpoint_number": checkpoint_number,
                            "throughput": value.get("throughput"),
                            "duration": value.get("duration"),
                            "start": value.get("start"),
                            "end": value.get("end"),
                        }

                        for nested_dict in [start_times, end_times, durations, throughputs, sizes]:
                            if not nested_dict[checkpoint_operation].get(checkpoint_number):
                                nested_dict[checkpoint_operation][checkpoint_number] = []

                        start_times[checkpoint_operation][checkpoint_number].append(datetime.strptime(value.get("start"), "%Y-%m-%dT%H:%M:%S.%f"))
                        end_times[checkpoint_operation][checkpoint_number].append(datetime.strptime(value.get("end"), "%Y-%m-%dT%H:%M:%S.%f"))
                        durations[checkpoint_operation][checkpoint_number].append(value.get("duration"))
                        throughputs[checkpoint_operation][checkpoint_number].append(value.get("throughput"))
                        sizes[checkpoint_operation][checkpoint_number].append(value.get("duration") * value.get("throughput"))

                        if not max_durations[checkpoint_operation].get(checkpoint_number) or value.get("duration") > max_durations[checkpoint_operation][checkpoint_number]:
                            # This is the maximum duration across ranks for a given checkpoint
                            max_durations[checkpoint_operation][checkpoint_number] = value.get("duration")
                        if checkpoint_number not in checkpoint_dicts[checkpoint_operation].keys():
                            checkpoint_dicts[checkpoint_operation][checkpoint_number] = []
                        checkpoint_dicts[checkpoint_operation][checkpoint_number].append(dict_to_append)

            self.metrics["checkpoint_size_GB"] = self.benchmark_result.summary['metric'].get("checkpoint_size_GB")

            op_max_durations = dict(save=list(), load=list())
            for operation in ("save", "load"):
                if max_durations[operation]:
                    # Report average of max durations
                    self.metrics[f"mean_of_max_{operation}_duration"] = mean(
                        [max_durations[operation][cp_num] for cp_num in max_durations[operation].keys()])

                    # Calculate the min starts and max ends across all ranks for a given checkpoint
                    min_starts = [min(s_times) for s_times in start_times[operation].values()]
                    max_ends = [max(e_times) for e_times in end_times[operation].values()]

                    calculated_max_duration[operation] = [max_end - min_start for max_end, min_start in zip(max_ends, min_starts)]
                    calculated_min_throughputs = [self.metrics["checkpoint_size_GB"] / dur.total_seconds() for dur in calculated_max_duration[operation]]
                    self.metrics[f"calculated_mean_{operation}_throughput_from_strict_times"] = mean(calculated_min_throughputs)


            min_throughputs = dict(save=list(), load=list())
            for operation, op_dicts in checkpoint_dicts.items():
                for checkpoint_number, data_dicts in op_dicts.items():
                    # Get minimum reported throughput from all ranks in a given checkpoint
                    min_throughputs[operation].append(min([stats.get("throughput") for stats in data_dicts]))

            for operation, throughput_list in min_throughputs.items():
                self.metrics[f"num_{operation}_ops"] = len(throughput_list)
                self.metrics[f"min_throughput_list_{operation}"] = throughput_list
                if not throughput_list:
                    continue
                self.metrics[f"mean_of_min_{operation}_throughput"] = mean(throughput_list)

        self.metrics["processes_per_host"] = self.num_processes / self.num_hosts



class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify call the self.check_* methods
    """
    def __init__(self, logger, checks=None, *args, **kwargs):
        self.logger = logger
        self.issues = []
        self.selected_checks = checks if checks else []
        
        # Dynamically find all check methods
        self.check_methods = [getattr(self, method) for method in dir(self) 
                             if callable(getattr(self, method)) and method.startswith('check_')]

        if self.selected_checks:
            self.check_methods = [method for method in self.check_methods if method.__name__ in self.selected_checks]
            self.logger.debug(f"Selected checks: {self.selected_checks} on {self.__class__.__name__} instance")
        
    def run_checks(self) -> List[Issue]:
        """Run all check methods and return a list of issues"""
        self.issues = []
        if self.selected_checks and not self.check_methods:
            # If checks are defined but none match, return an empty list
            return self.issues
        for check_method in self.check_methods:
            try:
                self.logger.debug(f"Running check '{check_method.__name__}'")
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

    # def check_system_description(self):
    #     issues = []
    #     self.logger.info(f'Checking system description: {self.benchmark_run.system_description}...')
    #     if self.benchmark_run.system_description_error:
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.INVALID,
    #             message=f"Error during system description parsing: {self.benchmark_run.system_description_error}",
    #             expected="Valid yaml",
    #             actual=self.benchmark_run.system_description_error,
    #             severity="error"
    #         ))
    #     if not self.benchmark_run.system_description:
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.INVALID,
    #             message="No system description found",
    #             expected="Description yaml",
    #             actual=self.benchmark_run.system_description,
    #             severity="error"
    #         ))
    #     if isinstance(self.benchmark_run.system_description, str):
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.INVALID,
    #             message=f"Error during system description parsing: {self.benchmark_run.system_description}",
    #             expected="Valid yaml",
    #             actual=self.benchmark_run.system_description,
    #             severity="error"
    #         ))
    #     return issues

    def _check_allowed_params(self, closed_allowed_params, open_allowed_params):
        """
        This method will verify that the only parameters that were set were the allowed parameters.
        Allowed for closed:
          - dataset.num_files_train
          - dataset.num_subfolders_train
          -
        :return:
        """
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

        if configured_num_files < (required_num_files * .99):
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient number of training files",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        elif configured_num_files == required_num_files:
            return Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Number of training files is exactly required number",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        elif configured_num_files > ( required_num_files * .99):
            return Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Number of training files is more than required number",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        
        return None

    def check_allowed_params(self) -> Optional[Union[List[Issue], Issue]]:
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
        issues = self._check_allowed_params(closed_allowed_params, open_allowed_params)
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

    def check_au_success(self) -> Optional[Issue]:
        try:
            if self.benchmark_run.model not in MODELS:
                # Only check  on training models
                return

            train_au_percent = self.benchmark_run.metrics.get('train_au_mean_percentage')

            if train_au_percent < AU_REQUIREMENT[self.benchmark_run.model]:
                return Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Training AU mean percentage is below {AU_REQUIREMENT[self.benchmark_run.model]}%, model: {self.benchmark_run.model}",
                    parameter="train_au_mean_percentage",
                    expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
                    actual=train_au_percent
                )
            else:
                return Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Training AU mean percentage is {AU_REQUIREMENT[self.benchmark_run.model]}% or above, model: {self.benchmark_run.model}",
                    parameter="train_au_mean_percentage",
                    expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
                    actual=train_au_percent
                )
        except Exception as e:
            import pdb
            pdb.set_trace()

    # def check_au_calculation(self):
    #     issues = []
    #     train_au_percent = self.benchmark_run.metrics.get('train_au_mean_percentage')
    #     train_mean_samples_per_second = self.benchmark_run.benchmark_result.summary['metric'].get("train_throughput_mean_samples_per_second")
    #     num_accelerators = self.benchmark_run.benchmark_result.summary['num_accelerators']
    #     sps_per_accelerator = train_mean_samples_per_second / num_accelerators
    #
    #     # Ideal AU comes from batch size, compute time
    #     workload_config = self.benchmark_run.benchmark_result.hydra_configs['config.yaml']['workload']
    #     compute_time = workload_config['train']['computation_time']
    #     batch_size = workload_config['reader']['batch_size']
    #
    #     idea_samples_per_second = batch_size / compute_time
    #     min_samples_per_second = idea_samples_per_second * AU_REQUIREMENT[self.benchmark_run.model] / 100
    #
    #     # how close does it need to be? half a percent
    #     closeness_factor = 0.995
    #
    #     calculated_au_percent = (sps_per_accelerator / idea_samples_per_second * 100)
    #     if calculated_au_percent < (AU_REQUIREMENT[self.benchmark_run.model] * closeness_factor):
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.INVALID,
    #             message=f"The calculated AU (from samples per second) is below the threshold",
    #             parameter="calculated_au_mean_percentage",
    #             expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
    #             actual=f"{calculated_au_percent:.2f}%",
    #         ))
    #     else:
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.CLOSED,
    #             message=f"The calculated AU (from samples per second) meets the threshold",
    #             parameter="calculated_au_mean_percentage",
    #             expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
    #             actual=f"{calculated_au_percent:.2f}%",
    #         ))
    #
    #     if abs(calculated_au_percent - train_au_percent) > (1 - closeness_factor) * 100:
    #         difference = (calculated_au_percent - train_au_percent) / train_au_percent * 100
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.INVALID,
    #             message=f"Calculated AU (from samples per second) is {difference:0.1f}% different from the training AU mean percentage",
    #             parameter="calculated_au_mean_percentage",
    #             expected=f"{train_au_percent:.2f}%",
    #             actual=f"{calculated_au_percent:.2f}%",
    #         ))
    #     else:
    #         issues.append(Issue(
    #             validation=PARAM_VALIDATION.CLOSED,
    #             message=f"Calculated AU (from samples per second) is the same as the training AU mean percentage",
    #             parameter="calculated_au_mean_percentage",
    #             expected=f"{train_au_percent:.2f}%",
    #             actual=f"{calculated_au_percent:.2f}%",
    #         ))
    #
    #     # if sps_per_accelerator < (min_samples_per_second * closeness_factor):
    #     #     issues.append(Issue(
    #     #         validation=PARAM_VALIDATION.INVALID,
    #     #         message=f"Training throughput is below {AU_REQUIREMENT[self.benchmark_run.model]}% per second, model: {self.benchmark_run.model}",
    #     #         parameter="train_throughput_mean_samples_per_second",
    #     #         expected=f">= {min_samples_per_second}",
    #     #         actual=sps_per_accelerator
    #     #     ))
    #     #     issues.append(Issue(
    #     #         validation=PARAM_VALIDATION.INVALID,
    #     #         message=f"The calculated AU (from samples per second) of ",
    #     #         parameter="calculated_au_mean_percentage",
    #     #         expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
    #     #         actual=(sps_per_accelerator / idea_samples_per_second * 100)
    #     #     ))
    #     # else:
    #     #     issues.append(Issue(
    #     #         validation=PARAM_VALIDATION.CLOSED,
    #     #         message=f"Training throughput is {AU_REQUIREMENT[self.benchmark_run.model]}% per second or above, model: {self.benchmark_run.model}",
    #     #         parameter="train_throughput_mean_samples_per_second",
    #     #         expected=f">= {min_samples_per_second}",
    #     #         actual=sps_per_accelerator
    #     #     ))
    #     #     issues.append(Issue(
    #     #         validation=PARAM_VALIDATION.CLOSED,
    #     #         message=f"Training throughput is below {AU_REQUIREMENT[self.benchmark_run.model]}% per second, model: {self.benchmark_run.model}",
    #     #         parameter="calculated_au_mean_percentage",
    #     #         expected=f">= {AU_REQUIREMENT[self.benchmark_run.model]}",
    #     #         actual=(sps_per_accelerator / idea_samples_per_second * 100)
    #     #     ))
    #     return issues

    def check_checkpoint_files_in_code(self) -> Optional[Issue]:
        pass

    def check_num_epochs(self) -> Optional[Issue]:
        pass

    def check_file_system_caching(self) -> Optional[Issue]:
        pass


class CheckpointingRunRulesChecker(RunRulesChecker):
    """Rules checker for checkpointing benchmarks"""

    def check_num_processes(self):
        issues = []
        category = self.benchmark_run.submitted_category
        num_processes = self.benchmark_run.benchmark_result.summary['num_accelerators']
        mode = self.benchmark_run.parameters['checkpoint'].get('mode')
        if category == CLOSED:
            if mode == "subset":
                if num_processes != 8:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.INVALID,
                        message=f"Expected 8 processes for subset checkpointing, but found {num_processes}",
                        parameter="num_processes",
                        expected="8",
                        actual=num_processes
                    ))
                elif num_processes == 8:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.CLOSED,
                        message=f"Expected 8 processes for subset checkpointing, and found {num_processes}",
                        parameter="num_processes",
                        expected="8",
                        actual=num_processes
                    ))
            elif mode:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Unexpected value for checkpointing mode: {mode}",
                    parameter="checkpoint.mode",
                    expected="subset or None",
                    actual=mode
                ))
            else:
                closed_processes = LLM_ALLOWED_VALUES[self.benchmark_run.model][3]
                if num_processes != closed_processes:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.INVALID,
                        message=f"Expected {closed_processes} processes for checkpointing, but found {num_processes}",
                        parameter="num_processes",
                        expected=closed_processes,
                        actual=num_processes
                    ))
                elif num_processes == closed_processes:
                    issues.append(Issue(
                        validation=PARAM_VALIDATION.CLOSED,
                        message=f"Expected {closed_processes} processes for checkpointing, and found {num_processes}",
                        parameter="num_processes",
                        expected=closed_processes,
                        actual=num_processes
                    ))
        elif category == OPEN:
            data_parallel_processes = LLM_ALLOWED_VALUES[self.benchmark_run.model][2]
            if num_processes % data_parallel_processes == 0:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Number of processes ({num_processes}) is a multiple of {data_parallel_processes}",
                    parameter="num_processes",
                    expected=num_processes,
                    actual=num_processes
                ))
            elif num_processes < data_parallel_processes:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Expected at least {data_parallel_processes} processes for OPEN checkpointing, but found {num_processes}",
                    parameter="num_processes",
                    expected=data_parallel_processes,
                    actual=num_processes
                ))

            elif num_processes % data_parallel_processes != 0:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Number of processes should be a multiple of {data_parallel_processes}, but found {num_processes}",
                    parameter="num_processes",
                    expected=f"{data_parallel_processes} multiple",
                    actual=num_processes
                ))
        return issues

    def check_allowed_params(self) -> Optional[Union[List[Issue], Issue]]:
        """
        This method will verify that the only parameters that were set were the allowed parameters.
        Allowed for closed:
          - dataset.num_files_train
          - dataset.num_subfolders_train
          -
        :return:
        """
        closed_allowed_params = ['checkpoint.checkpoint_folder', 'checkpoint.mode', 'checkpoint.num_checkpoints_read',
                                 'checkpoint.num_checkpoints_write', 'model.parallelism.data']
        open_allowed_params = []
        issues = self._check_allowed_params(closed_allowed_params, open_allowed_params)
        return issues

    def check_clear_cache_requirement(self):
        # The filesystem cache needs to be cleared before reads if memory is > 3x the checkpoint size
        issues = []

        # Get total memory size and amount of data in fileystem cache
        # Compare to size of checkpoint based on num_processes and LLM_SIZE_BY_RANK
        # Some host have a smaller checkpoint responsiblity so we need to look at the minimum
        model = self.benchmark_run.model
        num_processes = self.benchmark_run.benchmark_result.summary['num_accelerators']
        num_hosts = self.benchmark_run.benchmark_result.summary['num_hosts']
        host_mem_sizes = self.benchmark_run.benchmark_result.summary['host_memory_GB']
        per_rank_gb = calculate_checkpointing_size(model, num_processes, self.logger)

        # We're assuming round-robin rank placement. The first ranks will need more memory but they should
        # be distributed across all hosts
        # We'll create a map of hosts and a list of ranks to accumulate the memory requirements per host
        host_data = {i: {'total_mem': host_mem_sizes[i], 'rank_indexes': list(), 'total_checkpoint_gb': 0} for i in range(num_hosts)}

        for i, rank_gb in enumerate(per_rank_gb):
            host = i % num_hosts
            host_data[host]['rank_indexes'].append(i)
            host_data[host]['total_checkpoint_gb'] += per_rank_gb[i]

        for host, data in host_data.items():
            data['dataset_mem_multiplier'] = data['total_mem'] / data['total_checkpoint_gb']
            if data['total_mem'] > 3 * data['total_checkpoint_gb']:
                host_data[host]['clear_cache_required'] = True
            else:
                host_data[host]['clear_cache_required'] = False

        self.logger.info(f'RunID: {self.benchmark_run.run_id}: Checkpoint on Host0: {host_data[0]["total_checkpoint_gb"]}, Total Memon Host0: {host_data[0]["total_mem"]}')

        clear_cache_required = any({data['clear_cache_required'] for data in host_data.values()})

        num_reads = self.benchmark_run.parameters.get('checkpoint', {}).get('num_checkpoints_read', 0)
        num_writes = self.benchmark_run.parameters.get('checkpoint', {}).get('num_checkpoints_write', 0)
        if num_reads > 0 and num_writes > 0:
            if clear_cache_required:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Checkpointing reads and writes must be separated with caches cleared before reads",
                    parameter="checkpoint with separate reads and writes",
                    expected=f"Memory < {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                    actual=f"Memory == {host_data[0]['total_mem']}"
                ))
            if not clear_cache_required:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Checkpointing reads and writes can be run as a combined run without clearing caches",
                    parameter="checkpoint with separate reads and writes",
                    expected=f"Memory < {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                    actual=f"Memory == {host_data[0]['total_mem']}"
                ))

        elif num_reads >= 1:
            # We need to verify cache was cleared if required
            filesystem_cache_size_kB = int(self.benchmark_run.benchmark_result.summary['host_meminfo']['Cached'].split()[0])
            filesystem_cache_size_GB = filesystem_cache_size_kB / 1024 / 1024

            if clear_cache_required and filesystem_cache_size_GB > 3 * host_data[0]['total_checkpoint_gb']:
                issues.append(
                    Issue(
                        validation=PARAM_VALIDATION.INVALID,
                        message=f"Filesystem cache was not cleared before read operations.",
                        parameter="filesystem_cache_size_GB",
                        expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                        actual=f"{filesystem_cache_size_GB:.2f} GB",
                    )
                )
            elif clear_cache_required and filesystem_cache_size_GB <= 3 * host_data[0]['total_checkpoint_gb']:
                issues.append(
                    Issue(
                        validation=PARAM_VALIDATION.CLOSED,
                        message=f"Filesystem cache was cleared before read operations.",
                        parameter="filesystem_cache_size_GB",
                        expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                        actual=f"{filesystem_cache_size_GB:.2f} GB"
                    )
                )
            elif not clear_cache_required:
                issues.append(
                    Issue(
                        validation=PARAM_VALIDATION.CLOSED,
                        message=f"Clearing cache not required before read operations",
                        parameter="memory_capacity_to_dataset_size",
                        expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                        actual=f"{host_mem_sizes[0]:.2f} GB"
                    )
                )
        elif num_writes >= 1:
            issues.append(
                Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Filesystem cache clearing not required for write operations.",
                    parameter="clear_filesystem_cahe",
                    expected=None,
                    actual=None
                )
            )
        return issues


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
        for bench_run in self.benchmark_runs:
            if bench_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                num_writes += bench_run.parameters.get('checkpoint', {}).get('num_checkpoints_write', 0)
                num_reads += bench_run.parameters.get('checkpoint', {}).get('num_checkpoints_read', 0)

        reads = dict()
        writes = dict()
        combo = dict()

        for bench_run in self.benchmark_runs:
            if bench_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                write_GBs = bench_run.metrics.get('calculated_mean_save_throughput_from_strict_times', None)
                read_GBs = bench_run.metrics.get('calculated_mean_load_throughput_from_strict_times', None)
                if write_GBs:
                    writes[bench_run.run_id] = write_GBs
                if read_GBs:
                    reads[bench_run.run_id] = read_GBs
                if write_GBs and read_GBs:
                    combo[bench_run.run_id] = (write_GBs + read_GBs)/2

        # pdb.set_trace()
        if num_reads > 10:
            best_read_id = max(reads, key=reads.get)
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Find best read ID: {best_read_id} with {reads[best_read_id]} read GB/s",
                parameter="too many runs",
                expected=10,
                actual=num_reads
            ))

        if num_writes > 10:
            best_write_id = max(writes, key=writes.get)
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Find best write ID: {best_write_id} with {writes[best_write_id]} write GB/s",
                parameter="too many runs",
                expected=10,
                actual=num_writes
            ))

        if len(combo.keys()) > 1:
            best_combo_id = max(combo, key=combo.get)
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Find best combo ID: {best_combo_id} with {combo[best_combo_id]:.2f} average read and write GB/s",
                parameter="too many runs",
                expected=10,
                actual=num_reads + num_writes
            ))

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

    def check_filesystem_cache(self):
        # The filesystem cache needs to be cleared before reads if memory is > 3x the checkpoint size
        issues = []

        for benchmark_run in self.benchmark_runs:
            # Get total memory size and amount of data in fileystem cache
            # Compare to size of checkpoint based on num_processes and LLM_SIZE_BY_RANK
            # Some host have a smaller checkpoint responsiblity so we need to look at the minimum
            model = benchmark_run.model
            num_processes = benchmark_run.benchmark_result.summary['num_accelerators']
            num_hosts = benchmark_run.benchmark_result.summary['num_hosts']
            host_mem_sizes = benchmark_run.benchmark_result.summary['host_memory_GB']
            per_rank_gb = calculate_checkpointing_size(model, num_processes, self.logger)

            # We're assuming round-robin rank placement. The first ranks will need more memory but they should
            # be distributed across all hosts
            # We'll create a map of hosts and a list of ranks to accumulate the memory requirements per host
            host_data = {i: {'total_mem': host_mem_sizes[i], 'rank_indexes': list(), 'total_checkpoint_gb': 0} for i in range(num_hosts)}

            for i, rank_gb in enumerate(per_rank_gb):
                host = i % num_hosts
                host_data[host]['rank_indexes'].append(i)
                host_data[host]['total_checkpoint_gb'] += per_rank_gb[i]

            for host, data in host_data.items():
                data['dataset_mem_multiplier'] = data['total_mem'] / data['total_checkpoint_gb']
                if data['total_mem'] > 3 * data['total_checkpoint_gb']:
                    host_data[host]['clear_cache_required'] = True
                else:
                    host_data[host]['clear_cache_required'] = False

            clear_cache_required = any({data['clear_cache_required'] for data in host_data.values()})

            num_reads = benchmark_run.parameters.get('checkpoint', {}).get('num_checkpoints_read', 0)
            num_writes = benchmark_run.parameters.get('checkpoint', {}).get('num_checkpoints_write', 0)
            if num_reads >= 1:
                # We need to verify cache was cleared if required
                if benchmark_run.benchmark_result.per_rank_outputs:
                    # Look at the individual outputs and how much memory the host had in cache
                    # Not the default behavior, will do this later
                    continue

                filesystem_cache_size_kB = int(benchmark_run.benchmark_result.summary['host_meminfo']['Cached'].split()[0])
                filesystem_cache_size_GB = filesystem_cache_size_kB / 1024 / 1024

                if clear_cache_required and filesystem_cache_size_GB > 3 * host_data[0]['total_checkpoint_gb']:
                    issues.append(
                        Issue(
                            validation=PARAM_VALIDATION.INVALID,
                            message=f"Filesystem cache was not cleared before read operations.",
                            parameter="filesystem_cache_size_GB",
                            expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                            actual=f"{filesystem_cache_size_GB:.2f} GB",
                        )
                    )
                elif clear_cache_required and filesystem_cache_size_GB <= 3 * host_data[0]['total_checkpoint_gb']:
                    issues.append(
                        Issue(
                            validation=PARAM_VALIDATION.CLOSED,
                            message=f"Filesystem cache was cleared before read operations.",
                            parameter="filesystem_cache_size_GB",
                            expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                            actual=f"{filesystem_cache_size_GB:.2f} GB"
                        )
                    )
                elif not clear_cache_required:
                    issues.append(
                        Issue(
                            validation=PARAM_VALIDATION.CLOSED,
                            message=f"Clearing cache not required before read operations",
                            parameter="memory_capacity_to_dataset_size",
                            expected=f"<= {3 * host_data[0]['total_checkpoint_gb']:.1f} GB",
                            actual=f"{host_mem_sizes[0]:.2f} GB"
                        )
                    )

        return issues


class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    supported_models = MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 5 runs for training benchmarks
        """

        num_runs = 0
        for run in self.benchmark_runs:
            if run.benchmark_type == BENCHMARK_TYPES.training:
                num_runs += 1

        if num_runs == 5:
            return Issue(
                        validation=PARAM_VALIDATION.CLOSED,
                        message="Found expected 5 benchmark runs.",
                        severity="info",
                    )

        return Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Expected 5 training runs but found {num_runs}",
                    severity="error",
                )

    def _run_time_mean(self, runs_timestamps : List[Tuple[datetime, datetime, BenchmarkRun]]) -> timedelta:
        sum = timedelta(seconds=0)
        n = 0
        for start, end, _ in runs_timestamps:
            sum = sum + (end - start)
            n += 1
        return sum / n

    def check_inter_test_times(self) -> Optional[Issue]:
        # This can only operate on BenchmarkResults and not Benchmarks
        # Check if each benchmark_run has the benchmark_result attribute
        issues = []
        # Keep a tuple with the start and end time of each run, how long it took and the run object.
        runs_timestamps : List[Tuple[datetime, datetime, BenchmarkRun]] = []
        for run in self.benchmark_runs:
            self.logger.debug(f"Processing run {run.run_id}")
            if run.benchmark_result is not None:
                run_start_str = run.benchmark_result.summary.get("start", None)
                run_end_str = run.benchmark_result.summary.get("end", None)
                if run_start_str is None:
                    return Issue(
                                validation=PARAM_VALIDATION.INVALID,
                                message=f"Summary is missing start timestamp in run {run.run_id}.",
                                severity="error",
                            )
                elif run_end_str is None:
                    return Issue(
                                validation=PARAM_VALIDATION.INVALID,
                                message=f"Summary is missing end timestamp in run {run.run_id}.",
                                severity="error",
                            )
                try:
                    run_start = datetime.fromisoformat(run_start_str)
                except:
                    return Issue(
                                validation=PARAM_VALIDATION.INVALID,
                                message=f"Failed to parse start timestamp of run {run.run_id}.",
                                severity="error",
                            )
                try:
                    run_end = datetime.fromisoformat(run_end_str)
                except:
                    return Issue(
                                validation=PARAM_VALIDATION.INVALID,
                                message=f"Failed to parse end timestamp of run {run.run_id}.",
                                severity="error",
                            )
                runs_timestamps.append((run_start, run_end, run))
            else:
                # If the BenchmarkRun object does not have a BenchmarkResult, this rule is impossible to verify
                return Issue(
                            validation=PARAM_VALIDATION.INVALID,
                            message=f"Failed verify inter run times, benchmark {run.run_id} does not have results.",
                            severity="error",
                        )

        # Compute the mean computation time of each run. This will be used to check against the time between each run.
        # TODO (OMichaud0) This does not catch the case where someone might accept one bad long run in order to cherry-pick.
        mean_run_time = self._run_time_mean(runs_timestamps)

        # Sort the runs by start timestamp.
        runs_timestamps.sort(key=lambda run_ts: run_ts[0])

        prev_end = runs_timestamps[0][1]
        for start, end, run in runs_timestamps[1:]:
            if prev_end is not None and start - prev_end > mean_run_time:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Benchmark run {run.run_id} started too long after previous run",
                    parameter="inter_run_time",
                    expected=f"<= {mean_run_time.total_seconds():.2f} seconds",
                    actual=f"{(start - prev_end).total_seconds():.2f} seconds",
                    severity="error"
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Benchmark run {run.run_id} was within acceptable timeframe.",
                    parameter="inter_run_time",
                    expected=f"<= {mean_run_time.total_seconds():.2f} seconds",
                    actual=f"{(start - prev_end).total_seconds():.2f} seconds",
                    severity="info"
                ))
            prev_end = end
        return issues


class BenchmarkVerifier:

    def __init__(self, benchmark_runs, checks=None, logger=None):
        self.logger = logger
        self.issues = []
        self.checks = checks

        if isinstance(benchmark_runs, list):
            self.mode = "multi"
            self.benchmark_runs = benchmark_runs
        else:
            self.mode = "single"
            self.benchmark_runs = [benchmark_runs, ]

        if self.mode == "single":
            if "mlpstorage.benchmarks." in str(type(self.benchmark_runs[0])):
                # This is here if we get a Benchmark instance that needs to run the verifier
                # on itself before execution. We map it to a BenchmarkRun instance
                # We check against the string so we don't need to import the Benchmark classes here
                self.benchmark_runs = [BenchmarkRun(benchmark_instance=self.benchmark_runs[0], logger=logger)]

            benchmark_run = self.benchmark_runs[0]
            if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger, checks=self.checks)
            elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger, checks=self.checks)

        elif self.mode == "multi":
            benchmark_types = {br.benchmark_type for br in self.benchmark_runs}
            if len(benchmark_types) > 1:
                raise ValueError("Multi-run verification requires all runs are from the same benchmark type. Got types: {benchmark_types}")
            else:
                benchmark_type = benchmark_types.pop()

            if benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingSubmissionRulesChecker(benchmark_runs, logger, checks=self.checks)
            if benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointSubmissionRulesChecker(benchmark_runs, logger, checks=self.checks)

    def verify(self) -> PARAM_VALIDATION:
        run_ids = [br.run_id for br in self.benchmark_runs]
        if self.mode == "single":
            self.logger.status(f"Verifying benchmark run for {run_ids[0]}")
        elif self.mode == "multi":
            self.logger.status(f"Verifying benchmark runs for multiple runs including {run_ids[0]}")
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
            self.logger.status(f'Benchmark run is INVALID due to {num_invalid} issues ({run_ids[0]})')
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.INVALID
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.OPEN
            self.logger.status(f'Benchmark run qualifies for OPEN category ({run_ids[0]})')
            return PARAM_VALIDATION.OPEN
        else:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.CLOSED
            self.logger.status(f'Benchmark run qualifies for CLOSED category ({run_ids[0]})')
            return PARAM_VALIDATION.CLOSED


def calculate_checkpointing_size(model, num_processes, logger) -> List[float]:

    # Calculate the total writes per rank which equates to memory required per rank
    # If zero_level is 1, then rank 0 writes the entire model,
    # If zero_level is 3, then the model is sharded across all ranks
    min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES.get(model)
    model_gb, optimizer_gb = LLM_SIZE_BY_RANK.get(model)
    rank_gb = []

    logger.verbose(f'Model & optimizer size: {model_gb:.2f} GB, {optimizer_gb:.2f} GB')
    for rank in range(num_processes):
        rank_gb.append(0)
        if zero_level == 1:
            logger.ludicrous(
                "Optimizer is written by all ranks, but only the ranks on the first DP instance write the model")
            rank_gb[rank] = optimizer_gb / num_processes
            if rank < GPUpDP:
                rank_gb[rank] += model_gb / GPUpDP
                logger.ludicrous(f'First DP: rank-{rank} write model: {rank_gb[rank]:.2f} GB')
        elif zero_level == 3:
            rank_gb[rank] = (model_gb + optimizer_gb) / num_processes
            logger.ludicrous(f'Rank {rank} writes portion of model and optimizer: {rank_gb[rank]:.2f} GB')
        else:
            logger.error(f'Invalid zero_level: {zero_level}')
            raise ValueError("Invalid zero_level")

    return rank_gb


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


def get_runs_files(results_dir, submitters=None, exclude=None, logger=None):
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
    for root, dirs, files in os.walk(results_dir, topdown=True):
        # If we in the top directory and we have a list of submitters passed, only consider runs from those subdirectories
        if root == results_dir and submitters:
            dirs[:] = [d for d in dirs if d in submitters]
        if root == results_dir and exclude:
            dirs[:] = [d for d in dirs if d not in exclude]

        logger.ridiculous(f'Processing directory: {root}')

        # Look for metadata files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]

        if not metadata_files:
            logger.ridiculous(f'No metadata file found')
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

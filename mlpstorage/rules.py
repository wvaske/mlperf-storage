import abc
import enum
import os

from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint, pformat
from typing import List, Dict, Any, Optional, Tuple

from mlpstorage.config import (MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, BENCHMARK_TYPES,
                               DATETIME_STR, LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, HYDRA_OUTPUT_SUBDIR, BENCHMARK_TYPE)
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
        result = f"[{self.severity.upper()}] {self.message}"
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
    subcommand: str
    model: str
    run_datetime: str

    def __str__(self):
        id_str = self.program
        if self.command:
            id_str += f"_{self.command}"
        if self.subcommand:
            id_str += f"_{self.subcommand}"
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

    def __init__(self, host_info_list: List[str], logger):
        self.logger = logger
        self.host_info_list: host_info_list

        # Aggregated information across all hosts
        self.total_memory_bytes = 0
        self.total_cores = 0

        self.calculate_aggregated_info()

    def calculate_aggregated_info(self):
        """Calculate aggregated system information across all hosts"""
        for host_info in self.host_info_list:
            self.total_memory_bytes += host_info.memory.total
            self.total_cores += host_info.cpu.num_cores

    @classmethod
    def from_dlio_summary_json(cls, summary, logger) -> 'ClusterInformation':
        host_memories = summary.get("host_memory_GB")
        hosts = summary.get("hosts")
        host_info_list = []
        for i, host in enumerate(hosts):
            host_info = HostInfo(
                hostname=host,
                cpu=None,
                memory=HostMemoryInfo(total=host_memories[i] * 1024 * 1024 * 1024)
            )
            host_info_list.append(host_info)
        return cls(host_info_list, logger)


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
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                self.logger.verbose(f"Loaded DLIO summary from {summary_path}")
            except Exception as e:
                self.logger.error(f"Failed to load DLIO summary from {summary_path}: {e}")

        # Find and load Hydra config files if they exist
        hydra_dir = os.path.join(self.benchmark_result_root_dir, HYDRA_OUTPUT_SUBDIR)
        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            for config_file in os.listdir(hydra_dir):
                if config_file.endswith('.yaml'):
                    config_path = os.path.join(hydra_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            self.hydra_configs[config_file] = yaml.safe_load(f)
                        self.logger.verbose(f"Loaded Hydra config from {config_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Hydra config from {config_path}: {e}")


class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.
    Can be constructed either from a benchmark instance or from result files.
    """
    def __init__(self, logger, benchmark_result=None, benchmark_instance=None):
        self.logger = logger
        if benchmark_result is None and benchmark_instance is None:
            self.logger.error(f"The BenchmarkRun instance needs either a benchmark_result or a benchmark_instance.")
            raise ValueError("Either benchmark_result or benchmark_instance must be provided")
        if benchmark_result and benchmark_instance:
            self.logger.error(f"Both benchmark_result and benchmark_instance provided, which is not supported.")
            raise ValueError("Only one of benchmark_result and benchmark_instance can be provided")
            
        self.benchmark_type = None
        self.model = None
        self.command = None
        self.parameters = {}
        self.system_info = None
        self.metrics = {}
        self.run_id = None
        self.run_datetime = None
        
        if benchmark_instance:
            self._process_benchmark_instance(benchmark_instance)
        elif benchmark_result:
            self._process_benchmark_result(benchmark_result)

        self.run_id = RunID(program=self.benchmark_type, command=self.command, model=self.model,
                            run_datetime=self.run_datetime)

    def _process_benchmark_instance(self, benchmark_instance):
        """Extract parameters and system info from a running benchmark instance"""
        self.benchmark_type = benchmark_instance.BENCHMARK_TYPE
        self.model = getattr(benchmark_instance.args, 'model', None)
        self.command = getattr(benchmark_instance.args, 'command', None)
        self.run_datetime = benchmark_instance.run_datetime
        
        # Extract parameters from the benchmark instance
        if hasattr(benchmark_instance, 'combined_params'):
            self.parameters = benchmark_instance.combined_params
        else:
            # Fallback to args if combined_params not available
            self.parameters = vars(benchmark_instance.args)
            
        # Extract system information
        if hasattr(benchmark_instance, 'cluster_information'):
            self.system_info = benchmark_instance.cluster_information

    def _process_benchmark_result(self, benchmark_result):
        """Extract parameters and system info from result files"""
        # Process the summary and hydra configs to find what was run
        summary_workload = benchmark_result.summary.get('workload', {})
        summary_workflow = summary_workload.get('workflow', {})
        workflow = (
            summary_workflow.get('generate_data', {}),
            summary_workflow.get('train', {}),
            summary_workflow.get('checkpoint', {}),
        )

        # Get benchmark type based on workflow
        if workflow[0] or workflow[1]:
            self.benchmark_type = BENCHMARK_TYPES.training
        elif workflow[2]:
            self.benchmark_type = BENCHMARK_TYPES.checkpointing

        self.model = summary_workload.get('model', {}).get("name")

        if workflow[0] and not any(workflow[1], workflow[2]):
            self.command = "datagen"
        if workflow[1] and not any(workflow[0], workflow[2]):
            self.command = "run_benchmark"

        self.run_datetime = benchmark_result.summary.get("start")
        self.parameters = benchmark_result.hydra_conifigs.get("config.yaml", {}).get("workload", {})

        self.metrics = benchmark_result.summary.get("metric")
        self.system_info = ClusterInformation.from_dlio_summary_json(benchmark_result.summary, self.logger)


class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify benchmark runs against rules.
    """
    def __init__(self, benchmark_run, logger):
        self.benchmark_run = benchmark_run
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
    
    @abc.abstractmethod
    def check_benchmark_type(self) -> Optional[Issue]:
        """Check if the benchmark type is valid"""
        pass


class TrainingRulesChecker(RulesChecker):
    """Rules checker for training benchmarks"""
    
    def check_benchmark_type(self) -> Optional[Issue]:
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.training.name:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.training.name,
                actual=self.benchmark_run.benchmark_type
            )
        return None
    
    def check_model(self) -> Optional[Issue]:
        if not self.benchmark_run.model or self.benchmark_run.model not in MODELS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid or missing model: {self.benchmark_run.model}",
                parameter="model",
                expected=f"One of {MODELS}",
                actual=self.benchmark_run.model
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
        
        # For this example, we'll assume a minimum of 1000 files
        # In practice, you'd calculate this based on memory and other factors
        required_num_files = 1000
        
        if configured_num_files < required_num_files:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient number of training files",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        
        return None


class CheckpointingRulesChecker(RulesChecker):
    """Rules checker for checkpointing benchmarks"""
    def check_benchmark_type(self) -> Optional[Issue]:
        pass


class BenchmarkRunVerifier:

    def __init__(self, benchmark_run, logger):
        self.benchmark_run = benchmark_run
        self.logger = logger
        self.issues = []

        if self.benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
            self.rules_checker = TrainingRulesChecker(benchmark_run, logger)
        elif self.benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
            self.rules_checker = CheckpointingRulesChecker(benchmark_run, logger)

    def verify(self) -> PARAM_VALIDATION:
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

        if num_invalid > 0:
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            return PARAM_VALIDATION.OPEN
        else:
            return PARAM_VALIDATION.CLOSED


class BenchmarkVerifier:

    def __init__(self, benchmark, logger):
        self.benchmark = benchmark
        self.logger = logger

    def verify(self):
        # Training Verification
        if self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
            validation = self._verify_training_params()
        elif self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
            validation = self._verify_checkpointing_params()
        elif self.benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
            validation = self._verify_vector_database_params()
        else:
            validation = PARAM_VALIDATION.INVALID

        self.logger.status(f'Benchmark verification: {validation.name}')
        return validation

    def _verify_training_params(self) -> PARAM_VALIDATION:
        # Add code here for validation processes. We do not need to validate an option is in a list as the argparse
        #  option "choices" accomplishes this for us.

        # We will walk through all the params and see if they're valid for open, closed, or invalid.
        # Then we compare the set of validations against open/closed and exit if not a valid configuration.
        validation_results = dict()
        any_non_closed = False
        if self.benchmark.params_dict:
            for param, value in self.benchmark.params_dict.items():
                param_validation = self._verify_training_optional_param(self.benchmark.args.model, param, value)
                validation_results[param] = [self.benchmark.args.model, value, param_validation]
                if validation_results[param][2] != PARAM_VALIDATION.CLOSED:
                    any_non_closed = True

        self.logger.verbose(f'Verification results from input parames: \n{pformat(validation_results)}')
        # Accumulate the validations from optional params
        validation_set = set(v[2] for v in validation_results.values())


        # Add code to verify the other parameters here. Use cluster information and data size commands to verify
        cluster_info = self.benchmark.cluster_information
        total_client_mem_bytes = cluster_info.info['accumulated_mem_info_bytes']['total']
        num_hosts = len(cluster_info.info['host_info'])
        num_files, num_dirs, total_bytes = calculate_training_data_size(self.benchmark.args, cluster_info, self.benchmark.combined_params.get('dataset'),self.benchmark.combined_params.get('reader'), self.logger)

        # Verify num_files_train from combined_params is above the required minimum
        configured_num_files_train = int(self.benchmark.combined_params['dataset']['num_files_train'])
        if configured_num_files_train < num_files:
            self.logger.error(f'Configured number of files for training ({configured_num_files_train}) is less than required number of files ({num_files}).')
            if self.benchmark.args.command == "run":
                self.logger.error(f'Use the --param option to pass the correct number of files for training. "--param dataset.num_files_train=<number_of_files>"')
            validation_set.add(PARAM_VALIDATION.INVALID)
        else:
            self.logger.verbose(f'Configured number of files for training ({configured_num_files_train}) meets the required number of files ({num_files}).')
            validation_set.add(PARAM_VALIDATION.CLOSED)

        # TODO: Other verifications from the rules document should do validation_set.add(PARAM_VALIDATION)


        self.logger.verbose(f'Analyzing verification from set of results: {validation_set}')
        if validation_set == set():
            self.logger.error('Validation did not complete properly. Unable to verify close or open execution.')
            return PARAM_VALIDATION.INVALID
        elif validation_set == {PARAM_VALIDATION.CLOSED}:
            return PARAM_VALIDATION.CLOSED
        elif PARAM_VALIDATION.INVALID in validation_set:
            error_string = "\n\t".join([f"{p} = {v[1]}" for p, v in validation_results.items()])
            self.logger.error(f'Not all parameters allowed in closed submission: \n'
                              f'\t{error_string}')
            return PARAM_VALIDATION.INVALID
        else:
            # All open or closed:
            return PARAM_VALIDATION.OPEN

    def _verify_training_optional_param(self, model, param, value):
        if model in MODELS:
            # Allowed to change data_folder and number of files to train depending on memory requirements
            if param.startswith('dataset'):
                left, right = param.split('.')
                if right in ('data_folder', 'num_files_train'):
                    self.logger.verbose(f'Allowed to change {param} for model {model} with value {value}.')
                    return PARAM_VALIDATION.CLOSED

            # Allowed to set number of read threads
            if param.startswith('reader'):
                left, right = param.split('.')
                if right == "read_threads":
                    if 0 < int(value) < MAX_READ_THREADS_TRAINING:
                        self.logger.verbose(f'Allowed to change {param} for model {model} with value {value} being less than {MAX_READ_THREADS_TRAINING}.')
                        return PARAM_VALIDATION.CLOSED

        self.logger.error(f'Invalid parameter {param} for model {model} with value {value}.')
        return PARAM_VALIDATION.INVALID

    def _verify_checkpointing_params(self) -> PARAM_VALIDATION:
        # Rules to Implement:
        # Minimum of 4 processes per physical host during checkpointing
        # For closed, the number of processes can be exactly 8 (subset) or exactly TP x PP x DP from the config
        # For open, the number of processes can be a multiple of the TP x PP from the config

        model = self.benchmark.args.model
        min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES.get(model)
        num_procs = self.benchmark.args.num_processes
        num_hosts = len(self.benchmark.args.hosts)

        validations = set()
        if num_procs / num_hosts >= 4:
            validations.add(PARAM_VALIDATION.CLOSED)
            self.logger.verbose(f'Number of processes per host ({num_procs / num_hosts}) is at least 4.')
        else:
            self.logger.error(f'Number of processes per host ({num_procs / num_hosts}) should be at least 4.')
            validations.add(PARAM_VALIDATION.INVALID)

        if num_procs >= min_procs:
            validations.add(PARAM_VALIDATION.CLOSED)
            self.logger.verbose(f'Number of processes ({num_procs}) is at least {min_procs}.')
        else:
            self.logger.error(f'Number of processes ({num_procs}) should be at least {min_procs}.')
            validations.add(PARAM_VALIDATION.INVALID)

        if num_procs in [ClosedGPUs, LLM_SUBSET_PROCS]:
            self.logger.verbose(f'Number of processes ({num_procs}) is one of {LLM_SUBSET_PROCS} or {ClosedGPUs} in closed submission.')
            validations.add(PARAM_VALIDATION.CLOSED)
        elif self.benchmark.args.closed:
            self.logger.error(f'Number of processes ({num_procs}) should be exactly {LLM_SUBSET_PROCS} or {ClosedGPUs} in closed submission.')
            validations.add(PARAM_VALIDATION.INVALID)
        elif not benchmark.args.closed:
            # num procs should be a multiple of GPUpDP
            dp_instances = num_procs / GPUpDP
            if not dp_instances.is_integer():
                validations.add(PARAM_VALIDATION.INVALID)
                self.logger.error(f'Number of processes ({num_procs}) is not a multiple of {GPUpDP}.')
            else:
                # To get here we've already checked minimum procs, procs per host, and if closed is set
                validations.add(PARAM_VALIDATION.OPEN)
                self.logger.verbose(f'Number of processes ({num_procs}) is a multiple of {GPUpDP}.')

        if validations == {PARAM_VALIDATION.CLOSED}:
            return PARAM_VALIDATION.CLOSED
        elif PARAM_VALIDATION.INVALID in validations:
            return PARAM_VALIDATION.INVALID
        else:
            # Not only closed but no INVALID options == OPEN
            return PARAM_VALIDATION.OPEN


    def _verify_vector_database_params(self):
        # TODO: Implement validation for vector database parameters.
        # Use Cluster Information to verify the size of dataset against the number of clients?
        self.logger.info(f'Need to implement vector database parameter validation.')
        if self.benchmark.args.closed:
            self.logger.error(f'VectorDB is preview only and is not allowed in closed submission.')
            return PARAM_VALIDATION.INVALID

        return PARAM_VALIDATION.CLOSED


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger) -> Tuple[int, int, int]:
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
    measured_total_mem_bytes = cluster_information.info['accumulated_mem_info_bytes']['total']
    if args.client_host_memory_in_gb and args.num_client_hosts:
        # If host memory per client and num clients is provided, we use these values instead of the calculated memory
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
    elif args.clienthost_host_memory_in_gb and not args.num_client_hosts:
        # If we have memory but not clients, we use the number of provided hosts and given memory amount
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 102
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
    else:
        # If no args are provided, measure total memory for given hosts
        total_mem_bytes = measured_total_mem_bytes

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes
    file_size_bytes = dataset_params['num_samples_per_file'] * dataset_params['record_length_bytes']

    min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * args.num_processes * reader_params['batch_size']
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
        <command>:
            <subcommand> (Optional)
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
        <command>:
            <subcommand> (Optional)
                <datetime>:
                    <output files>

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


def get_runs_files(results_dir, benchmark_name=None, command=None, logger=None):
    """
    Walk the results_dir location and return a list of dictionaries that represent a single run

    [ { 'benchmark_name': <benchmark_name>,
      'command': <command>,
      'datetime': <datetime>,
      'mlps_metadata_file': <mlps_metadata_file_path>,
      'dlio_summary_json_file': <dlio_summary_json_file_path>,
      'run_number': run_<num>,  #(if applicable)
      'files': [<file_path1>, <file_path2>,...] } ]

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

        metadata_path = os.path.join(root, metadata_files[0])

        # Find DLIO summary.json file if it exists
        dlio_summary_file = None
        for f in files:
            if f == 'summary.json':
                dlio_summary_file = os.path.join(root, f)
                break

        # Collect all files in this run directory
        run_files = [os.path.join(root, f) for f in files]

        # Create run info dictionary
        run_info = {
            'mlps_metadata_file': metadata_path,
            'dlio_summary_json_file': dlio_summary_file,
            'files': run_files
        }

        runs.append(run_info)

    return runs


# MLPerf™ Storage V2.0 Benchmark Rules

## 1. General Rules (All Workloads)

### 1.1 Fairness and Ethics

#### 1.1.1 Fair Benchmarking
Benchmarking must be conducted to measure framework and storage system performance as fairly as possible.

#### 1.1.2 Replicability
Results that cannot be replicated are not valid results.

#### 1.1.3 Replication Tolerance
Replicated results must be within 5% within required tries.

### 1.2 System Availability

#### 1.2.1 Available Systems
All components of an "available system" must be publicly available.

#### 1.2.2 Unavailable Components
If components are not available at submission time, they must be included in the next round of submissions or results may be retracted.

#### 1.2.3 RDI Systems
RDI (research, development, internal) systems must be made available upon demand for replication by MLCommons.

### 1.3 Non-determinism

#### 1.3.1 Fixed Random Seed
The data generator uses a fixed random seed that must not be changed.

#### 1.3.2 Random Number Generator Sources
Random number generators may be seeded from: clock, system randomness (/dev/random, /dev/urandom), or another RNG initialized with an allowed seed.

#### 1.3.3 Storage System Isolation
The storage system must not be informed of the random seed or source of randomness.

### 1.4 Result Reporting

#### 1.4.1 Rounding
Public results must be rounded to two decimal places.

### 1.5 Persistent Storage

#### 1.5.1 Persistent Storage Requirement
All workloads must use persistent storage.

#### 1.5.2 Data Residency
Data must reside on persistent storage before benchmark testing begins.

### 1.6 Consecutive Runs Requirement

#### 1.6.1 Consecutive Execution
Multiple runs for a workload must be run consecutively.

#### 1.6.2 Inter-Run Timing
Time between runs (stop time to start time) must be less than the time to execute a single run.

#### 1.6.3 No Cherry-Picking
Cherry-picking of results is forbidden.

### 1.7 Submission Requirements

#### 1.7.1 YAML File Requirement
Each submission must map to a unique `<system-name>.yaml` file.

#### 1.7.2 PDF File Requirement
Each submission must map to a unique `<system-name>.pdf` file.

#### 1.7.3 Log Submission
All logs from every run must be submitted.

#### 1.7.4 Run Continuity
Runs must be consecutive with no failed runs between submitted runs.

## 2. Training Workload Rules

### 2.1 Supported Models

#### 2.1.1 3D U-Net
3D U-Net (image segmentation, medical) is a supported model.

#### 2.1.2 ResNet-50
ResNet-50 (image classification) is a supported model.

#### 2.1.3 CosmoFlow
CosmoFlow (cosmology parameter prediction) is a supported model.

### 2.2 Accelerator Emulation

#### 2.2.1 No Hardware Accelerators Required
Hardware accelerators (GPUs, TPUs, ASICs) are NOT required.

#### 2.2.2 DLIO Tool Usage
Benchmark emulates accelerators using DLIO tool.

#### 2.2.3 Sleep Call Replacement
Training on accelerator is replaced with a `sleep()` call.

#### 2.2.4 Sleep Interval Determination
Sleep interval depends on batch size and accelerator type.

#### 2.2.5 Supported Accelerator Types
Supported accelerator types: NVIDIA A100 and H100 GPUs.

### 2.3 Performance Metrics

#### 2.3.1 Primary Metric
Metric: samples per second.

#### 2.3.2 Minimum AU Thresholds
Must meet minimum Accelerator Utilization (AU) threshold:
- 3D U-Net: 90% AU minimum
- ResNet-50: 90% AU minimum
- CosmoFlow: 70% AU minimum

#### 2.3.3 AU Calculation
AU calculation: `(total_compute_time / ideal_benchmark_running_time) * 100`

### 2.3.4 Ideal Running Time Calculation
Ideal running time calculation: `((records_per_file * total_files) / simulated_accelerators) / (batch_size & computation_time * epochs)`

#### 2.3.5 First Step Exclusion
First step I/O operations excluded from AU calculation but included in samples/second.

### 2.4 Dataset Requirements

#### 2.4.1 Storage Location
Dataset must be in shared persistent storage at benchmark start.

#### 2.4.2 Dataset Size
Dataset size must be at least 5x the sum of memory across all MLPerf Storage nodes.

#### 2.4.3 Minimum Steps
Minimum 500 steps per epoch.

#### 2.4.4 Dataset Content
Dataset content must follow specifications for each model (see Table 1).

### 2.5 Dataset Generation

#### 2.5.1 Size Calculation Script
Must use MLPerf Storage benchmark script to calculate minimum dataset size.

#### 2.5.2 Synthetic Data
Synthetic data generated using DLIO.

#### 2.5.3 Generation Log Submission
Dataset generation logs must be included in submission.

### 2.6 Caching Rules

#### 2.6.1 Warm-up Run
All runs must use a warm-up run before the 5 test runs.

#### 2.6.2 Random Seed Variation
Random seed must change for each run (controlled by mlpstorage script).

### 2.7 Run Requirements

#### 2.7.1 Number of Runs
Must execute 5 consecutive runs.

#### 2.7.2 Final Metric Calculation
Final metric is the average across the 5 runs.

#### 2.7.3 Benchmark Termination
Benchmark ends after a predetermined number of epochs.

### 2.8 Single-Host Submissions

#### 2.8.1 Maximum Accelerators
Must include runs for maximum simulated accelerators on ONE HOST NODE.

#### 2.8.2 AU Threshold Maintenance
Must maintain above 90% AU threshold.

#### 2.8.3 Memory Requirement
Approximately 0.5GB host memory required per simulated accelerator.

### 2.9 Distributed Training Submissions

#### 2.9.1 Data Accessibility
All data must be accessible to all host nodes.

#### 2.9.2 Accelerator Uniformity
Number of simulated accelerators must be identical on each host node.

#### 2.9.3 Maximum Accelerators
Must include runs for maximum number of simulated accelerators across all hosts.

#### 2.9.4 AU Threshold Maintenance
Must maintain above 90% AU threshold.

#### 2.9.5 Parallelism Type
Only data parallelism supported (not model parallelism).

## 3. Checkpointing Workload Rules

### 3.1 Supported Models

#### 3.1.1 LLaMA 3 8B
LLaMA 3 8B (8 processes, 105 GB checkpoint) is a supported model.

#### 3.1.2 LLaMA 3 70B
LLaMA 3 70B (64 processes, 912 GB checkpoint) is a supported model.

#### 3.1.3 LLaMA 3 405B
LLaMA 3 405B (512 processes, 5.29 TB checkpoint) is a supported model.

#### 3.1.4 LLaMA 3 1T
LLaMA 3 1T (1024 processes, 18 TB checkpoint) is a supported model.

### 3.2 Accelerator Requirements

#### 3.2.1 No Hardware Accelerators Required
Hardware accelerators (GPUs, TPUs, ASICs) are NOT required.

#### 3.2.2 Accelerator Emulation
Benchmark emulates accelerators.

### 3.3 Operational Modes

#### 3.3.1 Default Mode - Purpose
Default mode is used for shared storage systems.

#### 3.3.2 Default Mode - Process Count
Total processes must match Table 2 (TP×PP×DP).

#### 3.3.3 Default Mode - Multi-Host
Runs on multiple hosts.

#### 3.3.4 Default Mode - Dataset Coverage
Writes/reads entire checkpoint dataset.

#### 3.3.5 Subset Mode - Purpose
Subset mode is intended for node local storage systems.

#### 3.3.6 Subset Mode - Process Count
Only 8 processes allowed (except 8B model which doesn't support subset mode).

#### 3.3.7 Subset Mode - Single Host
Runs on single host.

#### 3.3.8 Subset Mode - Dataset Coverage
Writes/reads fraction of checkpoint data.

### 3.4 Execution Sequence

#### 3.4.1 Step 1 - Write
Write 10 checkpoints.

#### 3.4.2 Step 2 - Clear Caches
Clear filesystem caches (if required).

#### 3.4.3 Step 3 - Read
Read 10 checkpoints.

### 3.5 Cache Clearing Requirements

#### 3.5.1 Cache Clearing Condition
Required if checkpoint size per client node < 3x client node memory capacity.

#### 3.5.2 Cache Clearing Timing
Must clear caches between write and read phases.

#### 3.5.3 Data Source Requirement
100% of read phase data must come from storage system, not filesystem cache.

#### 3.5.4 Cache Clearing Responsibility
Cache clearing performed outside mlpstorage tool.

### 3.6 fsync Requirement

#### 3.6.1 fsync Application
`fsync` must be applied during checkpoint writes.

#### 3.6.2 fsync Purpose
Ensures data is flushed to persistent storage.

#### 3.6.3 fsync Default
Enabled by default in all workload configuration files.

### 3.7 Performance Metrics

#### 3.7.1 Write Metric
Write bandwidth (throughput).

#### 3.7.2 Read Metric
Read bandwidth (throughput).

#### 3.7.3 Duration Metric
Duration metric: maximum time across all processes.

#### 3.7.4 Throughput Metric
Throughput metric: minimum across all processes.

### 3.8 Submission Requirements

#### 3.8.1 Write Checkpoints
Must include 10 checkpoints written.

#### 3.8.2 Read Checkpoints
Must include 10 checkpoints read.

#### 3.8.3 Optional Process Logs
Must include logs for any optional processes (cache clearing, storage remapping).

### 3.9 Simultaneous Read/Write Requirements

#### 3.9.1 Failure Scenario Simulation
Checkpoint recovery must mimic failure scenario where different hosts read than wrote.

#### 3.9.2 Remapping Time Measurement
For systems requiring "remapping" between write and read: time to remap must be measured and included.

#### 3.9.3 Remapping Time Addition
Duration between write completion and read availability added to recovery time.

#### 3.9.4 Remapping Documentation
Must be documented in SystemDescription.yaml.

### 3.10 Multi-Host Support Documentation

#### 3.10.1 Required YAML Fields
Required in system_configuration.yaml:
```yaml
System:
  shared_capabilities:
    multi_host_support: True/False
    simultaneous_write_support: True/False
    simultaneous_read_support: True/False
```
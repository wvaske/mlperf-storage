#!/usr/bin/env python3
"""
Milvus Vector Database Benchmark Script

This script executes random vector queries against a Milvus collection using multiple processes.
It measures and reports query latency statistics.
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import time
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import signal
import sys

try:
    from pymilvus import connections, Collection
except ImportError:
    print("Error: pymilvus package not found. Please install it with 'pip install pymilvus'")
    sys.exit(1)

# Global flag for graceful shutdown
shutdown_flag = mp.Value('i', 0)

def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shut down worker processes"""
    print("\nReceived interrupt signal. Shutting down workers gracefully...")
    with shutdown_flag.get_lock():
        shutdown_flag.value = 1

def read_disk_stats() -> Dict[str, Dict[str, int]]:
    """
    Read disk I/O statistics from /proc/diskstats
    
    Returns:
        Dictionary mapping device names to their read/write statistics
    """
    stats = {}
    try:
        with open('/proc/diskstats', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 14:  # Ensure we have enough fields
                    device = parts[2]
                    # Fields based on kernel documentation
                    # https://www.kernel.org/doc/Documentation/ABI/testing/procfs-diskstats
                    sectors_read = int(parts[5])  # sectors read
                    sectors_written = int(parts[9])  # sectors written
                    
                    # 1 sector = 512 bytes
                    bytes_read = sectors_read * 512
                    bytes_written = sectors_written * 512
                    
                    stats[device] = {
                        "bytes_read": bytes_read,
                        "bytes_written": bytes_written
                    }
        return stats
    except FileNotFoundError:
        print("Warning: /proc/diskstats not available (non-Linux system)")
        return {}
    except Exception as e:
        print(f"Error reading disk stats: {e}")
        return {}

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format with appropriate units"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    value = float(bytes_value)
    
    while value > 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
        
    return f"{value:.2f} {units[unit_index]}"

def calculate_disk_io_diff(start_stats: Dict[str, Dict[str, int]], 
                          end_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Calculate the difference in disk I/O between start and end measurements"""
    diff_stats = {}
    
    for device in end_stats:
        if device in start_stats:
            diff_stats[device] = {
                "bytes_read": end_stats[device]["bytes_read"] - start_stats[device]["bytes_read"],
                "bytes_written": end_stats[device]["bytes_written"] - start_stats[device]["bytes_written"]
            }
    
    return diff_stats

def generate_random_vector(dim: int) -> List[float]:
    """Generate a random normalized vector of the specified dimension"""
    vec = np.random.random(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()

def connect_to_milvus(host: str, port: str) -> bool:
    """Establish connection to Milvus server"""
    try:
        connections.connect(alias="default", host=host, port=port)
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def execute_batch_queries(
    process_id: int,
    host: str,
    port: str,
    collection_name: str,
    vector_dim: int,
    batch_size: int,
    report_count: int,
    max_queries: Optional[int],
    runtime_seconds: Optional[int],
    output_dir: str,
    shutdown_flag: mp.Value
) -> None:
    """
    Execute batches of vector queries and log results to disk
    
    Args:
        process_id: ID of the current process
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the collection to query
        vector_dim: Dimension of vectors
        batch_size: Number of queries to execute in each batch
        max_queries: Maximum number of queries to execute (None for unlimited)
        runtime_seconds: Maximum runtime in seconds (None for unlimited)
        output_dir: Directory to save results
        shutdown_flag: Shared value to signal process termination
    """
    # Connect to Milvus
    if not connect_to_milvus(host, port):
        return
    
    # Get collection
    try:
        collection = Collection(collection_name)
        collection.load()
    except Exception as e:
        print(f"Process {process_id}: Failed to load collection: {e}")
        return
    
    # Prepare output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"milvus_benchmark_p{process_id}_{timestamp}.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Track execution
    start_time = time.time()
    query_count = 0
    batch_count = 0
    
    print(f"Process {process_id}: Starting benchmark")
    
    try:
        with open(output_file, 'w') as f:
            while True:
                # Check if we should terminate
                with shutdown_flag.get_lock():
                    if shutdown_flag.value == 1:
                        break
                
                # Check termination conditions
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if runtime_seconds is not None and elapsed_time >= runtime_seconds:
                    break
                
                if max_queries is not None and query_count >= max_queries:
                    break
                
                # Generate batch of query vectors
                batch_vectors = [generate_random_vector(vector_dim) for _ in range(batch_size)]
                
                # Execute batch and measure time
                batch_start = time.time()
                try:
                    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
                    results = collection.search(
                        data=batch_vectors,
                        anns_field="vector",
                        param=search_params,
                        limit=10,
                        output_fields=["id"]
                    )
                    batch_end = time.time()
                    batch_success = True
                except Exception as e:
                    print(f"Process {process_id}: Search error: {e}")
                    batch_end = time.time()
                    batch_success = False
                
                # Record batch results
                batch_time = batch_end - batch_start
                batch_count += 1
                query_count += batch_size
                
                # Log batch results to file
                batch_data = {
                    "process_id": process_id,
                    "batch_id": batch_count,
                    "timestamp": current_time,
                    "batch_size": batch_size,
                    "batch_time_seconds": batch_time,
                    "avg_query_time_seconds": batch_time / batch_size,
                    "success": batch_success
                }
                
                f.write(json.dumps(batch_data) + "\n")
                f.flush()  # Ensure data is written to disk immediately
                
                # Print progress
                if batch_count % report_count == 0:
                    print(f"Process {process_id}: Completed {query_count} queries in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Process {process_id}: Error during benchmark: {e}")
    
    finally:
        # Disconnect from Milvus
        try:
            connections.disconnect("default")
        except:
            pass
        
        print(f"Process {process_id}: Finished. Executed {query_count} queries in {time.time() - start_time:.2f} seconds")

def calculate_statistics(results_dir: str) -> Dict[str, float]:
    """Calculate statistics from benchmark results"""
    all_latencies = []
    
    # Read all result files
    for file_path in Path(results_dir).glob("milvus_benchmark_*.jsonl"):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data["success"]:
                        # Convert to milliseconds for readability
                        query_time_ms = data["avg_query_time_seconds"] * 1000
                        # Add each query in the batch individually
                        all_latencies.extend([query_time_ms] * data["batch_size"])
                except Exception as e:
                    print(f"Error parsing result file {file_path}: {e}")
    
    if not all_latencies:
        return {"error": "No valid query results found"}
    
    # Calculate statistics
    latencies = np.array(all_latencies)
    total_queries = len(latencies)
    total_time_seconds = sum(latencies) / 1000  # Convert back to seconds for throughput
    
    stats = {
        "total_queries": total_queries,
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "p999_latency_ms": float(np.percentile(latencies, 99.9)),
        "p9999_latency_ms": float(np.percentile(latencies, 99.99)),
        "throughput_qps": float(total_queries / total_time_seconds) if total_time_seconds > 0 else 0
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Milvus Vector Database Benchmark")
    
    # Required parameters
    parser.add_argument("--processes", type=int, required=True, help="Number of parallel processes")
    parser.add_argument("--batch-size", type=int, required=True, help="Number of queries per batch")
    parser.add_argument("--vector-dim", type=int, required=True, help="Vector dimension")
    parser.add_argument("--report-count", type=int, default=10, help="Number of queries between logging results")
    
    # Database parameters
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--collection", type=str, required=True, help="Collection name to query")
    
    # Termination conditions (at least one must be specified)
    termination_group = parser.add_argument_group("termination conditions (at least one required)")
    termination_group.add_argument("--runtime", type=int, help="Maximum runtime in seconds")
    termination_group.add_argument("--queries", type=int, help="Total number of queries to execute")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                        help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Validate termination conditions
    if args.runtime is None and args.queries is None:
        parser.error("At least one termination condition (--runtime or --queries) must be specified")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create output directory
    benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"milvus_benchmark_{benchmark_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save benchmark configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "processes": args.processes,
        "batch_size": args.batch_size,
        "report_count": args.report_count,
        "vector_dim": args.vector_dim,
        "host": args.host,
        "port": args.port,
        "collection": args.collection,
        "runtime_seconds": args.runtime,
        "total_queries": args.queries
    }
    
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting benchmark with {args.processes} processes")
    print(f"Results will be saved to: {results_dir}")
    
    # Read initial disk stats
    start_disk_stats = read_disk_stats()
    
    # Calculate queries per process if total queries specified
    max_queries_per_process = None
    if args.queries is not None:
        max_queries_per_process = args.queries // args.processes
        # Add remainder to the first process
        remainder = args.queries % args.processes
    
    # Start worker processes
    processes = []
    try:
        for i in range(args.processes):
            # Adjust queries for the first process if there's a remainder
            process_max_queries = None
            if max_queries_per_process is not None:
                process_max_queries = max_queries_per_process + (remainder if i == 0 else 0)
            
            p = mp.Process(
                target=execute_batch_queries,
                args=(
                    i,
                    args.host,
                    args.port,
                    args.collection,
                    args.vector_dim,
                    args.batch_size,
                    args.report_count,
                    process_max_queries,
                    args.runtime,
                    results_dir,
                    shutdown_flag
                )
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        # Signal all processes to terminate
        with shutdown_flag.get_lock():
            shutdown_flag.value = 1
        
        # Wait for processes to terminate
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
    
    # Read final disk stats
    end_disk_stats = read_disk_stats()
    
    # Calculate disk I/O during benchmark
    disk_io_diff = calculate_disk_io_diff(start_disk_stats, end_disk_stats)
    
    # Calculate and print statistics
    print("\nCalculating benchmark statistics...")
    stats = calculate_statistics(results_dir)
    
    # Save statistics to file
    with open(os.path.join(results_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Total Queries: {stats.get('total_queries', 0)}")
    print(f"Mean Latency: {stats.get('mean_latency_ms', 0):.2f} ms")
    print(f"Median Latency: {stats.get('median_latency_ms', 0):.2f} ms")
    print(f"95th Percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
    print(f"99th Percentile: {stats.get('p99_latency_ms', 0):.2f} ms")
    print(f"99.9th Percentile: {stats.get('p999_latency_ms', 0):.2f} ms")
    print(f"Throughput: {stats.get('throughput_qps', 0):.2f} queries/second")
    
    # Print disk I/O statistics
    print("\nDISK I/O DURING BENCHMARK")
    print("-"*50)
    if disk_io_diff:
        # Calculate totals across all devices
        total_bytes_read = sum(dev_stats["bytes_read"] for dev_stats in disk_io_diff.values())
        total_bytes_written = sum(dev_stats["bytes_written"] for dev_stats in disk_io_diff.values())
        
        print(f"Total Bytes Read: {format_bytes(total_bytes_read)}")
        print(f"Total Bytes Written: {format_bytes(total_bytes_written)}")
        print("\nPer-Device Breakdown:")
        
        for device, io_stats in disk_io_diff.items():
            bytes_read = io_stats["bytes_read"]
            bytes_written = io_stats["bytes_written"]
            if bytes_read > 0 or bytes_written > 0:  # Only show devices with activity
                print(f"  {device}:")
                print(f"    Read:  {format_bytes(bytes_read)}")
                print(f"    Write: {format_bytes(bytes_written)}")
    else:
        print("Disk I/O statistics not available")
    
    print("\nDetailed results saved to:", results_dir)
    print("="*50)

if __name__ == "__main__":
    main()
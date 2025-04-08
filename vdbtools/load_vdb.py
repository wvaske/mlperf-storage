#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Vector Database Benchmark Tool")
    # Existing arguments
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--collection-name", type=str, default="benchmark_collection", help="Collection name")
    parser.add_argument("--dimension", type=int, default=1536, help="Vector dimension")
    parser.add_argument("--num-vectors", type=int, default=1000000, help="Number of vectors to generate")
    parser.add_argument("--distribution", type=str, default="normal", choices=["normal", "uniform"],
                        help="Vector distribution")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for insertion")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards for the collection")
    parser.add_argument("--monitor-interval", type=int, default=10,
                        help="Interval in seconds to check index building progress")
    parser.add_argument("--vector-dtype", type=str, default="float32", choices=["float16", "float32"],
                        help="Vector data type")
    parser.add_argument("--force", action="store_true", help="Force drop existing collection if it exists")

    # Keep index_type and metric_type as CLI args
    parser.add_argument("--index-type", type=str, default="DISKANN", help="Index type")
    parser.add_argument("--metric-type", type=str, default="COSINE", help="Metric type (COSINE, L2, IP)")

    # Add M and efConstruction as CLI args
    parser.add_argument("--M", type=int, default=64, help="DISKANN M parameter (graph degree)")
    parser.add_argument("--ef-construction", type=int, default=200, help="DISKANN efConstruction parameter")

    return parser.parse_args()

def connect_to_milvus(host, port):
    """Connect to Milvus server"""
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port,
            max_receive_message_length=514_983_574,
            max_send_message_length=514_983_574
        )
        logger.info(f"Connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False

def create_collection(collection_name, dim, num_shards, vector_dtype, force=False):
    """Create a new collection with the specified parameters"""
    try:
        # Check if collection exists
        if utility.has_collection(collection_name):
            if force:
                Collection(name=collection_name).drop()
                logger.info(f"Dropped existing collection: {collection_name}")
            else:
                logger.warning(f"Collection '{collection_name}' already exists. Use --force to drop and recreate it.")
                return None

        # Define vector data type
        vector_type = DataType.FLOAT_VECTOR

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=vector_type, dim=dim)
        ]
        schema = CollectionSchema(fields, description="Benchmark Collection")

        # Create collection
        collection = Collection(name=collection_name, schema=schema, num_shards=num_shards)
        logger.info(f"Created collection '{collection_name}' with {dim} dimensions and {num_shards} shards")

        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return None

def generate_vectors(num_vectors, dim, distribution='uniform'):
    """Generate random vectors based on the specified distribution"""
    if distribution == 'uniform':
        vectors = np.random.random((num_vectors, dim)).astype('float32')
    elif distribution == 'normal':
        vectors = np.random.normal(0, 1, (num_vectors, dim)).astype('float32')
    elif distribution == 'zipfian':
        # Simplified zipfian-like distribution
        base = np.random.random((num_vectors, dim)).astype('float32')
        skew = np.random.zipf(1.5, (num_vectors, 1)).astype('float32')
        vectors = base * (skew / 10)
    else:
        vectors = np.random.random((num_vectors, dim)).astype('float32')

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    return normalized_vectors.tolist()

def insert_data(collection, vectors, batch_size=10000):
    """Insert vectors into the collection in batches"""
    total_vectors = len(vectors)
    num_batches = (total_vectors + batch_size - 1) // batch_size

    start_time = time.time()
    total_inserted = 0

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total_vectors)
        batch_size_actual = batch_end - batch_start

        # Prepare batch data
        ids = list(range(batch_start, batch_end))
        batch_vectors = vectors[batch_start:batch_end]

        # Insert batch
        try:
            collection.insert([ids, batch_vectors])
            total_inserted += batch_size_actual

            # Log progress
            progress = total_inserted / total_vectors * 100
            elapsed = time.time() - start_time
            rate = total_inserted / elapsed if elapsed > 0 else 0

            logger.info(f"Inserted batch {i+1}/{num_batches}: {progress:.2f}% complete, "
                       f"rate: {rate:.2f} vectors/sec")

        except Exception as e:
            logger.error(f"Error inserting batch {i+1}: {str(e)}")

    # Flush the collection
    flush_start = time.time()
    logger.info(f"Inserted {total_inserted} vectors in {time.time() - start_time:.2f} seconds. Flushing...")
    collection.flush()
    flush_time = time.time() - flush_start
    logger.info(f"Flush completed in {flush_time:.2f} seconds")

    return total_inserted, time.time() - start_time

def create_index(collection, index_params):
    """Create an index on the collection"""
    try:
        start_time = time.time()
        logger.info(f"Creating index with parameters: {index_params}")
        collection.create_index("vector", index_params)
        index_creation_time = time.time() - start_time
        logger.info(f"Index creation command completed in {index_creation_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        return False

def monitor_index_building(collection_name, monitor_interval=60):
    """Monitor the progress of index building"""
    start_time = time.time()

    try:
        # Get initial progress
        prev_progress = utility.index_building_progress(collection_name=collection_name)
        prev_check_time = start_time
        initial_indexed_rows = prev_progress.get("indexed_rows", 0)
        initial_pending_rows = prev_progress.get("pending_index_rows", 0)
        total_rows = prev_progress.get("total_rows", 0)

        logger.info(f"Starting to monitor index building for collection: {collection_name}")
        logger.info(f"Initial state: {initial_indexed_rows:,} of {total_rows:,} rows indexed")
        logger.info(f"Initial pending rows: {initial_pending_rows:,}")

        # Track the phases of indexing
        indexing_phase_complete = False
        pending_phase_complete = False if initial_pending_rows > 0 else True
        compaction_performed = False
        
        # Track time with zero pending rows
        pending_zero_start_time = None
        pending_zero_threshold = 15  # 5 minutes in seconds

        while True:
            time.sleep(monitor_interval)  # Check at specified interval
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_check = current_time - prev_check_time
            
            try:
                progress = utility.index_building_progress(collection_name=collection_name)
                
                # Calculate progress metrics
                indexed_rows = progress.get("indexed_rows", 0)
                total_rows = progress.get("total_rows", total_rows)  # Use previous if not available
                pending_rows = progress.get("pending_index_rows", 0)
                
                # Calculate both overall and recent indexing rates
                total_rows_indexed_since_start = indexed_rows - initial_indexed_rows
                rows_since_last_check = indexed_rows - prev_progress.get("indexed_rows", indexed_rows)
                
                # Calculate pending rows reduction
                pending_rows_reduction = prev_progress.get("pending_index_rows", pending_rows) - pending_rows
                pending_reduction_rate = pending_rows_reduction / time_since_last_check if time_since_last_check > 0 else 0
                
                # Calculate overall rate (based on total time since monitoring began)
                if elapsed_time > 0 and total_rows_indexed_since_start > 0:
                    overall_indexing_rate = total_rows_indexed_since_start / elapsed_time  # rows per second
                    remaining_rows = total_rows - indexed_rows
                    estimated_seconds_remaining = remaining_rows / overall_indexing_rate if overall_indexing_rate > 0 else float('inf')
                    
                    # Alternative estimate based on pending rows
                    pending_estimate = pending_rows / pending_reduction_rate if pending_reduction_rate > 0 and pending_rows > 0 else float('inf')
                    
                    # Calculate recent rate (for comparison)
                    recent_indexing_rate = rows_since_last_check / time_since_last_check if time_since_last_check > 0 else 0
                    
                    # Calculate percent done
                    percent_done = indexed_rows / total_rows * 100 if total_rows > 0 else 0
                    
                    # Format the estimated time remaining
                    eta = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                    eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Format the pending-based estimate
                    pending_eta = datetime.now() + timedelta(seconds=pending_estimate) if pending_estimate != float('inf') else "Unknown"
                    if isinstance(pending_eta, datetime):
                        pending_eta_str = pending_eta.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        pending_eta_str = str(pending_eta)
                    
                    # Check if indexing phase is complete
                    if not indexing_phase_complete and indexed_rows >= total_rows:
                        indexing_phase_complete = True
                        logger.info(f"Initial indexing phase complete! All {indexed_rows:,} rows have been indexed.")
                        
                        # Perform compaction after indexing phase completes
                        if not compaction_performed:
                            logger.info("Performing collection compaction...")
                            collection = Collection(name=collection_name)
                            collection.compact()
                            compaction_performed = True
                            logger.info("Compaction completed.")
                        
                        logger.info(f"Now processing {pending_rows:,} pending rows...")
                        
                        # If pending rows is 0 after indexing completes, start the timer
                        if pending_rows == 0 and pending_zero_start_time is None:
                            pending_zero_start_time = current_time
                            logger.info("No pending rows detected. Starting 5-minute confirmation timer.")
                    
                    # Log progress with estimates
                    if not indexing_phase_complete:
                        # Still in initial indexing phase
                        logger.info(
                            f"Phase 1 - Building index: {percent_done:.2f}% complete... "
                            f"({indexed_rows:,}/{total_rows:,} rows) | "
                            f"Pending rows: {pending_rows:,} | "
                            f"Overall rate: {overall_indexing_rate:.2f} rows/sec | "
                            f"Recent rate: {recent_indexing_rate:.2f} rows/sec | "
                            f"ETA: {eta_str} | "
                            f"Est. remaining: {timedelta(seconds=int(estimated_seconds_remaining))}"
                        )
                    else:
                        # In pending rows processing phase
                        if pending_rows > 0:
                            # Reset the zero pending timer if we see pending rows
                            pending_zero_start_time = None
                            
                            logger.info(
                                f"Phase 2 - Processing pending rows: {pending_rows:,} remaining | "
                                f"Reduction rate: {pending_reduction_rate:.2f} rows/sec | "
                                f"ETA: {pending_eta_str} | "
                                f"Est. remaining: {timedelta(seconds=int(pending_estimate)) if pending_estimate != float('inf') else 'Unknown'}"
                            )
                        else:
                            # If pending rows is 0, check if we need to start the timer
                            if pending_zero_start_time is None:
                                pending_zero_start_time = current_time
                                logger.info("No pending rows detected. Starting 5-minute confirmation timer.")
                            else:
                                # Check if we've waited long enough with zero pending rows
                                zero_pending_time = current_time - pending_zero_start_time
                                logger.info(f"No pending rows for {zero_pending_time:.1f} seconds (waiting for 5 minutes to confirm)")
                                
                                if zero_pending_time >= pending_zero_threshold:
                                    logger.info("No pending rows detected for 5 minutes. Index building is considered complete.")
                                    pending_phase_complete = True
                else:
                    # If no progress since monitoring began
                    percent_done = indexed_rows / total_rows * 100 if total_rows > 0 else 0
                    logger.info(
                        f"Building index: {percent_done:.2f}% complete... "
                        f"({indexed_rows:,}/{total_rows:,} rows) | "
                        f"Pending rows: {pending_rows:,} | "
                        f"No significant progress detected yet"
                    )
                
                # Check if pending phase is complete
                if not pending_phase_complete and pending_rows == 0:
                    # If we've already waited 5 minutes with zero pending rows
                    if pending_zero_start_time is not None and (current_time - pending_zero_start_time) >= pending_zero_threshold:
                        pending_phase_complete = True
                        logger.info(f"Pending rows processing complete! All pending rows have been processed.")
                
                # Check if both phases are complete
                if (indexed_rows >= total_rows or indexing_phase_complete) and pending_phase_complete:
                    total_time = time.time() - start_time
                    logger.info(f"Index building fully complete! Total time: {timedelta(seconds=int(total_time))}")
                    break
                    
                # Update for next iteration
                prev_progress = progress
                prev_check_time = current_time
                
            except Exception as e:
                logger.error(f"Error checking index progress: {str(e)}")
                time.sleep(5)  # Short delay before retrying
                
    except Exception as e:
        logger.error(f"Error in monitor_index_building: {str(e)}")
        return False

def main():
    args = parse_args()

    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        return 1

    # Determine vector data type
    try:
        # Check if FLOAT16 is available in newer versions of pymilvus
        if hasattr(DataType, 'FLOAT16'):
            vector_dtype = DataType.FLOAT16 if args.vector_dtype == 'float16' else DataType.FLOAT_VECTOR
        else:
            # Fall back to supported data types
            logger.warning("FLOAT16 data type not available in this version of pymilvus. Using FLOAT_VECTOR instead.")
            vector_dtype = DataType.FLOAT_VECTOR
    except Exception as e:
        logger.warning(f"Error determining vector data type: {str(e)}. Using FLOAT_VECTOR as default.")
        vector_dtype = DataType.FLOAT_VECTOR

    # Create collection
    collection = create_collection(
        collection_name=args.collection_name,
        dim=args.dimension,
        num_shards=args.num_shards,
        vector_dtype=vector_dtype
    )

    if collection is None:
        return 1

    # Generate vectors
    logger.info(
        f"Generating {args.num_vectors} vectors with {args.dimension} dimensions using {args.distribution} distribution")
    start_gen_time = time.time()
    vectors = generate_vectors(args.num_vectors, args.dimension, args.distribution)
    gen_time = time.time() - start_gen_time
    logger.info(f"Generated {args.num_vectors} vectors in {gen_time:.2f} seconds")

    # Insert data
    logger.info(f"Inserting {args.num_vectors} vectors into collection '{args.collection_name}'")
    total_inserted, insert_time = insert_data(collection, vectors, args.batch_size)
    logger.info(f"Inserted {total_inserted} vectors in {insert_time:.2f} seconds")

    # Create index with updated parameters
    index_params = {
        "index_type": args.index_type,
        "metric_type": args.metric_type,
        "params": {
            "M": args.M,
            "efConstruction": args.ef_construction
        }
    }

    if not create_index(collection, index_params):
        return 1

    # Monitor index building
    logger.info(f"Starting to monitor index building progress (checking every {args.monitor_interval} seconds)")
    monitor_index_building(args.collection_name, args.monitor_interval)

    # Summary
    logger.info("Benchmark completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
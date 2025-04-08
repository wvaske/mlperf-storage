from pymilvus import connections, Collection, utility
import time
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530", 
                   max_receive_message_length=514_983_574, 
                   max_send_message_length=514_983_574)
collection_name = "diskann_openai"
collection = Collection(name=collection_name)

start_time = time.time()
prev_progress = utility.index_building_progress(collection_name=collection_name)
prev_check_time = start_time
initial_indexed_rows = prev_progress.get("indexed_rows", 0)
initial_pending_rows = prev_progress.get("pending_index_rows", 0)
total_rows = prev_progress.get("total_rows", 0)

logging.info(f"Starting to monitor index building for collection: {collection_name}")
logging.info(f"Initial state: {initial_indexed_rows:,} of {total_rows:,} rows indexed")
logging.info(f"Initial pending rows: {initial_pending_rows:,}")

# Track the phases of indexing
indexing_phase_complete = False
pending_phase_complete = False if initial_pending_rows > 0 else True

while True:
    time.sleep(60)  # Check every minute
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
                logging.info(f"Initial indexing phase complete! All {indexed_rows:,} rows have been indexed.")
                logging.info(f"Now processing {pending_rows:,} pending rows...")
            
            # Log progress with estimates
            if not indexing_phase_complete:
                # Still in initial indexing phase
                logging.info(
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
                    logging.info(
                        f"Phase 2 - Processing pending rows: {pending_rows:,} remaining | "
                        f"Reduction rate: {pending_reduction_rate:.2f} rows/sec | "
                        f"ETA: {pending_eta_str} | "
                        f"Est. remaining: {timedelta(seconds=int(pending_estimate)) if pending_estimate != float('inf') else 'Unknown'}"
                    )
        else:
            # If no progress since monitoring began
            percent_done = indexed_rows / total_rows * 100 if total_rows > 0 else 0
            logging.info(
                f"Building index: {percent_done:.2f}% complete... "
                f"({indexed_rows:,}/{total_rows:,} rows) | "
                f"Pending rows: {pending_rows:,} | "
                f"No significant progress detected yet"
            )
        
        # Check if pending phase is complete
        if not pending_phase_complete and pending_rows == 0:
            pending_phase_complete = True
            logging.info(f"Pending rows processing complete! All pending rows have been processed.")
        
        # Check if both phases are complete
        if (indexed_rows >= total_rows or indexing_phase_complete) and (pending_rows == 0 or pending_phase_complete):
            total_time = time.time() - start_time
            logging.info(f"Index building fully complete! Total time: {timedelta(seconds=int(total_time))}")
            break
            
        # Update for next iteration
        prev_progress = progress
        prev_check_time = current_time
        
    except Exception as e:
        logging.error(f"Error checking index progress: {str(e)}")
        time.sleep(5)  # Short delay before retrying
from pymilvus import connections, Collection
import time
import numpy as np

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530", max_receive_message_length=514_983_574, max_send_message_length=514983574)
collection_name = "diskann_openai"
collection = Collection(name=collection_name)

# Function to Generate Random Data
def generate_data(num_vectors, dim=1536, id_start=0):
    ids = list(range(num_vectors))
    vectors = np.random.random((num_vectors, dim)).tolist()
    return [ids, vectors]


# Insert in batches
def insert_in_batches(collection, data, batch_size=1000):
    for i in range(0, len(data[0]), batch_size):
        batch = [field[i:i + batch_size] for field in data]
        collection.insert(batch)


print(f'Begin inserting 100M vectors in 1M batches...')
start_time = time.time()
# Insert Data for Different Sizes
cur_id_start = 0
batch_size = 1000
for i in range(1):  # 1M, 10M, 100M
    size = 1000000
    print(f"Generating {i+1} iteration {size} vectors...")
    start_gen_time = time.time()
    data = generate_data(size, id_start=cur_id_start)
    cur_id_start += size
    print(f"Generated {size} vectors in {time.time() - start_gen_time:.2f} seconds. Doing insert...")
    # collection.insert(data)
    start_insert_time = time.time()
    print(f'Inserting vectors in batches of {batch_size}...')
    insert_in_batches(collection, data, batch_size=batch_size)
    print(f"Inserted {size} vectors in {time.time() - start_insert_time:.2f} seconds.")


start_flush_time = time.time()
print(f'Inserted 100M vectors in {time.time() - start_time:.2f} seconds. Flushing...')
collection.flush()
print(f'Flushed in {time.time() - start_flush_time:.2f} seconds.')


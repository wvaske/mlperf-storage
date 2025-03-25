from pymilvus import connections, Collection
import numpy as np

# Connect to Milvus
connections.connect("default", host="localhost", port="19530", max_receive_message_length=514983574, max_send_message_length=514983574)
collection_name = "diskann_openai"
collection = Collection(name=collection_name)

# Function to Generate Random Data
def generate_data(num_vectors, dim=1536):
    ids = list(range(num_vectors))
    vectors = np.random.random((num_vectors, dim)).tolist()
    return [ids, vectors]


# Insert in batches
def insert_in_batches(collection, data, batch_size=1000):
    for i in range(0, len(data[0]), batch_size):
        batch = [field[i:i + batch_size] for field in data]
        collection.insert(batch)


# Insert Data for Different Sizes
for size in [1_000_000, 10_000_000, 100_000_000]:  # 1M, 10M, 100M
    print(f"Inserting {size} vectors...")
    data = generate_data(size)
    # collection.insert(data)
    insert_in_batches(collection, data, batch_size=1000)
    collection.flush()
    print(f"Inserted {size} vectors.")

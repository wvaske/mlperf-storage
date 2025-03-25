from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# Connect to Milvus
connections.connect("default", host="localhost", port="19530", max_receive_message_length=514983574, max_send_message_length=514983574)

# Create Collection Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields, "DiskANN Benchmark Collection")

# Create Collection
collection_name = "diskann_openai"
collection = Collection(name=collection_name, schema=schema)

# Create DiskANN Index
index_params = {
    "index_type": "DISKANN",
    "metric_type": "COSINE",
    "params": {"M": 64, "efConstruction": 200},
}
collection.create_index("vector", index_params)
print(f"Collection '{collection_name}' created with DiskANN index.")

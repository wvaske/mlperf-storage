import yaml
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def load_config():
    with open("config.yml", "r") as config_file:
        return yaml.safe_load(config_file)

def load_workloads():
    with open("workloads.yml", "r") as workloads_file:
        return yaml.safe_load(workloads_file)["workloads"]


def connect_to_milvus(host, port):
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")


def create_collection(collection_name, dim):
    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Benchmark Collection")

    # Create or get the collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection {collection_name} created with dimension {dim}.")
    return collection


def insert_data(collection, dataset_size, dim):
    print(f"Inserting {dataset_size} vectors into the collection...")
    # Generate random vectors
    vectors = np.random.random((dataset_size, dim)).tolist()

    # Insert data
    collection.insert([vectors])
    collection.flush()
    print(f"Inserted {dataset_size} vectors.")



def perform_search(collection, query_size, dim, search_params):
    print(f"Performing search with {query_size} query vectors...")
    # Generate random query vectors
    query_vectors = np.random.random((query_size, dim)).tolist()

    # Perform search
    results = collection.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=10,
        expr=None  # Optional filter expression
    )

    # Print search results
    for i, result in enumerate(results):
        print(f"Query {i}: {len(result)} results")


def run_workloads(config, workloads):
    # Connect to Milvus
    connect_to_milvus(config["milvus"]["host"], config["milvus"]["port"])

    for workload in workloads:
        print(f"Running workload: {workload['name']}")

        # Create or load the collection
        collection = create_collection(
            collection_name=config["milvus"]["collection_name"],
            dim=config["milvus"]["dim"]
        )

        # Insert data
        insert_data(
            collection=collection,
            dataset_size=workload["dataset_size"],
            dim=config["milvus"]["dim"]
        )
        
        # Load Collection
        collection.load()

        # Perform search
        search_params = {"ef": workload["search_params"]["ef"]}
        perform_search(
            collection=collection,
            query_size=workload["query_size"],
            dim=config["milvus"]["dim"],
            search_params=search_params
        )

        print(f"Completed workload: {workload['name']}")

if __name__ == "__main__":
    # Load configurations and workloads
    config = load_config()
    workloads = load_workloads()

    # Run the workloads
    run_workloads(config, workloads)



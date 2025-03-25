This has been tested on Python3.11.
## Setup Milvus VectorDB and VectorDBBench

After cloning this repository, please do the following:

### Install Dependencies
```
sudo apt  install docker-compose
git clone https://github.com/milvus-io/milvus.git
```

### Modify or replace the docker-compose.yml file from this repo in Milvus repo (please edit the path to local storage directories in docker yml file)
### Start Milvus
```
cp vectordbScripts/docker-compose.yml milvus/deployments/docker/standalone/docker-compose.yml
sudo docker-compose up -d
```


### List of all running containers to verify setup  and Install VectorDBBench
```
sudo docker container ls -a
```

```
ssgroot@test84:~/DI_VectorDB_Tests$ sudo docker container ls -a
CONTAINER ID   IMAGE                                      COMMAND                  CREATED       STATUS                  PORTS                                                                                      NAMES
96764f0bb8c1   milvusdb/milvus:v2.5.0-beta                "/tini -- milvus run…"   10 days ago   Up 22 hours (healthy)   0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-standalone
eb556e566978   minio/minio:RELEASE.2023-03-20T20-16-18Z   "/usr/bin/docker-ent…"   10 days ago   Up 22 hours (healthy)   0.0.0.0:9000-9001->9000-9001/tcp, :::9000-9001->9000-9001/tcp                              milvus-minio
4262da84c397   quay.io/coreos/etcd:v3.5.14                "etcd -advertise-cli…"   10 days ago   Up 22 hours (healthy)   2379-2380/tcp                                                                              milvus-etcd
```
```
pip3.11 install vectordb-bench
```

If issue this happens:
```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

Do the following:
```
python3.11 -m pip install --upgrade setuptools wheel twine check-wheel-contents
```
For:

```
error: metadata-generation-failed
```

Do the following:

```
pip3.11 install setuptools==68  --> downgrading setuptools works
pip3.11 install vectordb-bench
```

### Running VectorDBBench with the DiskANN Index

1. Setup Milvus, Configure VectorDBBench
2. Prepare Dataset and Insert Data
3. Running Benchmark

```
python3.11 setup_milvus.py
python3.11 insert_data.py
python3.11 search_benchmark.py
```

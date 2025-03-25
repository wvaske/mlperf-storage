#!/usr/bin/env python3

import argparse
import numpy
import os
import pandas

from pyarrow import parquet, Table


def parse_args():
    parser = argparse.ArgumentParser(description='Parse command line arguments')

    parser.add_argument("--num-files", type=int, default=1,
                        help="Number of files to generate")
    parser.add_argument("--data-dimension", type=int, default=1536,
                        help="Dimensionality of the data")
    parser.add_argument("--samples-per-file", type=int, default=10,
                        help="Number of samples per file")

    parser.add_argument("--distribution", choices=["uniform", "normal"], default="uniform",
                        help="Distribution of the parameters in each vector")

    parser.add_argument("--data-dir", type=str, default="/tmp/vdb-data",
                        help="Directory to store generated data results")

    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return {k: v for k, v in vars(parser.parse_args()).items()}


def generate_data(num_files, data_dimension, samples_per_file, distribution, data_dir, *args, **kwargs):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(num_files + 1):
        filename = f"train-{i}-of-{num_files}.parquet"
        filepath = os.path.join(data_dir, filename)

        idx = numpy.arange(samples_per_file)

        if distribution == "uniform":
            data = numpy.random.uniform(size=(samples_per_file, data_dimension))
        elif distribution == "normal":
            data = numpy.random.normal(size=(samples_per_file, data_dimension))
        else:
            raise ValueError("Invalid distribution specified")

        data_list = data.tolist()
        df = pandas.DataFrame(dict(id=idx, emb=data_list))
        table = Table.from_pandas(df)
        parquet.write_table(table, filepath)


def main():
    args = parse_args()
    generate_data(**args)


if __name__ == "__main__":
    main()
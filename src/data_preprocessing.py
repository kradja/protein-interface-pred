import os
import pdb
import pickle

import pandas as pd
import polars as pl


def convert_data(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f, encoding="latin")
    return data


def write_parquet(data, output_file):
    pdb.set_trace()
    # rr = [{"r_vertex":x['r_vertex'],"l_vertex":x["l_vertex"],"label":x["label"]} for x in data[1]]
    print("End")

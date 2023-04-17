import itertools
import os
import pdb
import pickle
import random

import numpy as np
import pandas as pd
import polars as pl


def convert_data(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f, encoding="latin")
    return data


def write_parquet(data, output_file,neg_ratio=50):
    """There is a 70 length vector of features for each amino acid within a protein structure"""
    features_label = []
    for ind, complex_ind in enumerate(data[0]):
        ligand_aa_features = data[1][ind]["l_vertex"]
        receptor_aa_features = data[1][ind]["r_vertex"]
        aa_combinations = set(
            itertools.product(
                range(np.shape(ligand_aa_features)[0]),
                range(np.shape(receptor_aa_features)[0]),
            )
        )
        aa_label = data[1][ind]["label"]
        pdb.set_trace()
        mask = aa_label[:, 2] == 1
        aa_truth_label = aa_label[mask, :]
        aa_truth_label = set(map(tuple, aa_truth_label[:, (0, 1)]))
        aa_neg_label = random.sample(
            list(aa_combinations - aa_truth_label), len(aa_truth_label) * neg_ratio
        )
        features_label.extend(
            [
                np.concatenate(
                    (ligand_aa_features[x[0]], receptor_aa_features[x[1]], 1),
                    axis=None,
                )
                for x in aa_truth_label
            ]
        )
        features_label.extend(
            [
                np.concatenate(
                    (ligand_aa_features[x[0]], receptor_aa_features[x[1]], 0), axis=None
                )
                for x in aa_neg_label
            ]
        )
    df = pd.DataFrame(features_label, dtype="float32[pyarrow]")
    df.to_parquet(output_file, compression="gzip", index=False)
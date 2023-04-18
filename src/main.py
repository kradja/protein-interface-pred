import argparse
import os
import json

import data_preprocessing as dp
import models

# Data Location and Paper Implementation
# https://github.com/fouticus/pipgcn/tree/fd5f12c26fcabf4934d0a27fbc5a6753e0910fcd
# https://zenodo.org/record/1127774#.ZBD7fy-B0UR

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def write_json(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f)

def main(args):
    params = {"raw": "../data/00_raw_data/", "processed": "../data/01_processed/"}
    if args.preprocess:
        train = dp.convert_data(os.path.join(params["raw"], "train.cpkl"))
        test = dp.convert_data(os.path.join(params["raw"], "test.cpkl"))
        complex_train_files = dp.write_tensors(train, params["processed"],"complex_train")
        complex_test_files = dp.write_tensors(test, params["processed"],"complex_test")

        # Write the complex files to json
        write_json(complex_train_files, os.path.join(params["processed"], "complex_train.json"))
        write_json(complex_test_files, os.path.join(params["processed"], "complex_test.json"))
    if args.running:
        complex_train_files = read_json(os.path.join(params["processed"], "complex_train.json"))
        complex_test_files = read_json(os.path.join(params["processed"], "complex_test.json"))
        models.run_gcn(complex_train_files, complex_test_files)
        print("hey")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-preprocess",
        action="store_true",
    )
    parser.add_argument(
        "-running",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)

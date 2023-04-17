import argparse
import os

import data_preprocessing as dp

# Data Location and Paper Implementation
# https://github.com/fouticus/pipgcn/tree/fd5f12c26fcabf4934d0a27fbc5a6753e0910fcd
# https://zenodo.org/record/1127774#.ZBD7fy-B0UR


def main(args):
    params = {"raw": "../data/00_raw_data/", "processed": "../data/01_processed/"}
    train_file= os.path.join(params["processed"], "train.parquet.gzip")
    test_file= os.path.join(params["processed"], "test.parquet.gzip")
    if args.preprocess:
        train = dp.convert_data(os.path.join(params["raw"], "train.cpkl"))
        test = dp.convert_data(os.path.join(params["raw"], "test.cpkl"))
        dp.write_parquet(train, train_file)
        dp.write_parquet(test, test_file)
        print("convert")
    if args.testing:
        print("hey")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-preprocess",
        action="store_true",
    )
    parser.add_argument(
        "-testing",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)

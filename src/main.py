import argparse
import os

import data_preprocessing as dp

# Data Location and Paper Implementation
# https://github.com/fouticus/pipgcn/tree/fd5f12c26fcabf4934d0a27fbc5a6753e0910fcd
# https://zenodo.org/record/1127774#.ZBD7fy-B0UR


def main(args):
    params = {"raw": "../data/00_raw_data/", "processed": "../data/01_processed/"}
    if args.preprocess:
        train = dp.convert_data(os.path.join(params["raw"], "train.cpkl"))
        test = dp.convert_data(os.path.join(params["raw"], "test.cpkl"))
        dp.write_parquet(train, os.path.join(params["processed"], "train.parquet"))
        dp.write_parquet(test, os.path.join(params["processed"], "test.parquet"))
        print("convert")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-preprocess",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)

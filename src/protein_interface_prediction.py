#!/usr/src/env python
import argparse
import yaml

from prediction import prediction
from evaluation import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Protein interface prediction pipeline')
    parser.add_argument('-c', '--config', required=True,
                        help="File containing configuration to execute the pipeline.\n")
    args = parser.parse_args()
    return args


# Returns a config map for the yaml at the path specified
def parse_config(config_file_path):
    config = None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as err:
        print(f"Error parsing config file: {err}")
    return config


def main():
    args = parse_args()
    config = parse_config(args.config)
    config_type = config["config_type"]

    if config_type == "prediction":
        prediction.execute(config)
    elif config_type == "evaluation":
        evaluation.execute(config)
    else:
        print("ERROR: Unsupported configuration for config_type. Supported values=data_preprocessor", "classification", "evaluation")
    return


if __name__ == '__main__':
    main()

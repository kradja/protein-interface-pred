import os
import pandas as pd
from pathlib import Path
from src.evaluation.binary_classification_evaluation import BinaryClassEvaluation


def execute(config):
    input_settings = config["input_settings"]
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_evaluation_dir = output_settings["evaluation_dir"]
    output_visualization_dir = output_settings["visualization_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]

    evaluation_settings = config["evaluation_settings"]

    df = read_inputs(input_settings)

    evaluation_output_file_base_path = os.path.join(output_dir, output_evaluation_dir, output_dataset_dir)
    visualization_output_file_base_path = os.path.join(output_dir, output_visualization_dir, output_dataset_dir)
    # create any missing parent directories
    Path(os.path.dirname(evaluation_output_file_base_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(visualization_output_file_base_path)).mkdir(parents=True, exist_ok=True)

    evaluation_executor = BinaryClassEvaluation(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_prefix)
    evaluation_executor.execute()
    return


def read_inputs(input_settings):
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]

    inputs = []
    for key, file_name in input_file_names.items():
        input_file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(input_file_path)
        print(f"input file = {input_file_path} --> results size = {df.shape}")
        df["experiment"] = key
        inputs.append(df)
    df = pd.concat(inputs)
    return df


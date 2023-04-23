import pickle
import torch
import os
from pathlib import Path
import pandas as pd


def get_device(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def read_dataset(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin")
    return data


def write_output(model_dfs, output_dir, output_filename_prefix, output_type):
    for model_name, dfs in model_dfs.items():
        output_file_name = f"{output_filename_prefix}{model_name}_{output_type}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        # 5. Write the classification output
        print(f"Writing {output_type} of {model_name} to {output_file_path}")
        pd.concat(dfs).to_csv(output_file_path, index=False)


#def compute_loss()
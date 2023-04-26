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


def bce_focal_loss(bce_loss, label, alpha=0.9, gamma = 2):
    """ Binary Cross Entropy focal loss, mean.
    bce_loss: Binary Cross Entropy loss, a torch tensor.
    label: truth, 0s and 1s.
    gamma: hyperparameter, a float scalar, 2
    alpha: weight for distribution of classes, greater than 0.5 for more negative classes
    better than weighted loss for high degree of class balance. Emphasizes importance of
    examples with low predicted probability. Downweights easy examples.
    https://arxiv.org/pdf/1708.02002.pdf
    """
    p_t = torch.exp(-bce_loss) #predicted prob to transform to prob distribution
    alpha_tensor = alpha * label + (1 - alpha) * (1 - label)
    focal_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return torch.mean(focal_loss)

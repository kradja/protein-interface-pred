import os
import pandas as pd
import random

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import kmer_utils, utils
from prediction.models import logistic_regression
from prediction import prediction_with_presplit_inputs


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_files = input_settings["file_names"]
    input_files = [os.path.join(input_dir, input_file) for input_file in input_files]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    # classification settings
    classification_settings = config["classification_settings"]
    n = classification_settings["n_iterations"]
    models = classification_settings["models"]

    # 1. Read the data files
    df = read_dataset(input_files)

    # 2. filter out noise: labels configured to be excluded, NaN labels
    df = utils.filter_noise(df, label_settings)

    # 3. Compute kmer features
    kmer_df = kmer_utils.compute_kmer_features(df, k, label_col)

    # 4. Group the labels (if applicable) and convert the string labels to mapped integer indices
    kmer_df_with_transformed_label, idx_label_map = utils.transform_labels(kmer_df, classification_type, label_settings)

    # 5. Perform classification
    for model in models:
        if model["active"] is False:
            print(f"Skipping {model['name']} ...")
            continue
        model_name = model["name"]
        output_file_name = f"kmer_k{k}_{model_name}_{label_col}_{classification_type}_tr{train_proportion}_n{n}" + output_prefix + "_output.csv"
        output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

        # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
        model["n"] = n
        model["train_proportion"] = train_proportion
        model["label_col"] = label_col
        model["classification_type"] = classification_type

        if model["name"] == "lr":
            print("Executing Logistic Regression")
            results_df = execute_lr_classification(kmer_df_with_transformed_label, model)
        else:
            continue
        # Remap the class indices to original input labels
        results_df.rename(columns=idx_label_map, inplace=True)
        results_df["y_true"] = results_df["y_true"].map(idx_label_map)

        # 5. Write the classification output
        print(f"Writing results of {model_name} to {output_file_path}")
        results_df.to_csv(output_file_path, index=False)


def read_dataset(input_files, label_col):
    datasets = []
    for input_file in input_files:
        df = pd.read_csv(input_file, usecols=["id", "sequence", label_col])
        print(f"input file: {input_file}, size = {df.shape}")
        datasets.append(df)

    dataset = pd.concat(datasets)
    dataset.set_index("id", inplace=True)
    print(f"Size of input dataset = {dataset.shape}")
    return dataset


def execute_lr_classification(df, model):
    results = []
    for i in range(model["n"]):
        print(f"Iteration {i}")
        X_train, X_test, y_train, y_test = create_splits(df, model["train_proportion"], model["label_col"])
        y_pred = logistic_regression.run(X_train, X_test, y_train, model)
        result_df = pd.DataFrame(y_pred)
        result_df["itr"] = i
        result_df["y_true"] = y_test

        print(f"result size = {result_df.shape}")
        results.append(result_df)
    return pd.concat(results)


def create_splits(df, train_proportion, label_col):
    seed = random.randint(0, 10000)
    print(f"seed={seed}")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label_col]).values, df[label_col].values,
                                                        train_size=train_proportion, random_state=seed, stratify=df[label_col].values)

    # Standardize dataset
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    print(f"X_train size = {X_train.shape}")
    print(f"X_test size = {X_test.shape}")
    print(f"y_train size = {y_train.shape}")
    print(f"y_test size = {y_test.shape}")
    return X_train, X_test, y_train, y_test

import os
import pandas as pd
import random

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.prediction.models.baseline import logistic_regression
from src.prediction.models.baseline import random_forest
from src.utils import utils


def execute(input_settings, output_settings, classification_settings):
    # input settings
    input_dir = input_settings["input_dir"]
    input_files = input_settings["file_names"]
    training_dataset_files = [os.path.join(input_dir, input_file) for input_file in input_files["train"]]
    testing_dataset_files = [os.path.join(input_dir, input_file) for input_file in input_files["test"]]

    # output settings
    output_dir = output_settings["output_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix + "_" if output_prefix is not None else ""

    # classification settings
    n = classification_settings["n_iterations"]
    models = classification_settings["models"]

    # 1. Read the training dataset files
    train_df = read_dataset(training_dataset_files)
    test_df = read_dataset(training_dataset_files)

    # combine train and test df to get different splits for iterations
    df = pd.concat([train_df, test_df])

    # 2. Perform classification
    results = {}
    for itr in range(n):
        X_train, X_test, y_train, y_test = create_splits(df, classification_settings["train_proportion"], classification_settings["label_col"])
        for model in models:
            if model["active"] is False:
                print(f"Skipping {model['name']} ...")
                continue
            model_name = model["name"]
            if model_name not in results:
                # first iteration
                results[model_name] = []

            # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
            model["n"] = n

            if model["name"] == "lr":
                print("Executing Logistic Regression")
                y_pred = logistic_regression.run(X_train, X_test, y_train, model)
            elif model["name"] == "rf":
                print("Executing Random Forest")
                y_pred = random_forest.run(X_train, X_test, y_train, model)
            else:
                continue
            results[model_name].append(pd.DataFrame({"y_pred": y_pred[:, 1], "y_true": y_test, "model": model_name, "itr": itr}))

    # 5. Write the classification output
    utils.write_output(results, output_dir, output_prefix, "output")


def read_dataset(input_files):
    datasets = []
    for input_file in input_files:
        df = pd.read_csv(input_file)
        print(f"input file: {input_file}, size = {df.shape}")
        datasets.append(df)

    dataset = pd.concat(datasets)
    print(f"Size of input dataset = {dataset.shape}")
    return dataset


def create_splits(df, train_proportion, label_col):
    seed = random.randint(0, 10000)
    print(f"seed={seed}")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label_col]).values, df[label_col].values,
                                                        train_size=train_proportion, random_state=seed,
                                                        stratify=df[label_col].values)

    # Standardize dataset
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    print(f"X_train size = {X_train.shape}")
    print(f"X_test size = {X_test.shape}")
    print(f"y_train size = {y_train.shape}")
    print(f"y_test size = {y_test.shape}")
    return X_train, X_test, y_train, y_test

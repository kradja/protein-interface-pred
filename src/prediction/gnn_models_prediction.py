import os
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn as nn

from src.prediction.models.gnn.pdb_db_dataset import PDB_DB_Dataset
from src.prediction.models.gnn.gcn_ff import GCN_FFN
from src.prediction.models.gnn.gat_ff import GAT_FFN
from src.utils import utils


def execute(input_settings, output_settings, classification_settings):
    # input settings
    input_dir = input_settings["input_dir"]
    # Pipeline expects the following folder structure
    # <input_dir>
    # ----train
    #     ----graph files
    #     ----protein_complex_ids.txt
    #     ----protein_interface_labels.csv
    # ----test
    #     ----graph files
    #     ----protein_complex_ids.txt
    #     ----protein_interface_labels.csv

    # output settings
    output_dir = output_settings["output_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix + "_" if output_prefix is not None else ""

    # classification settings
    n = classification_settings["n_iterations"]
    models = classification_settings["models"]

    # 1. Read the training dataset files
    training_dataset = PDB_DB_Dataset(dirpath=os.path.join(input_dir, "train"))
    testing_dataset = PDB_DB_Dataset(dirpath=os.path.join(input_dir, "test"))

    # 2. Perform classification
    results = {}
    for itr in range(n):
        # load one pair of graphs and all the interface pairs within them in one batch
        train_data_loader = DataLoader(training_dataset, batch_size=1)
        test_data_loader = DataLoader(testing_dataset, batch_size=1)
        for model in models:
            if model["active"] is False:
                print(f"Skipping {model['name']} ...")
                continue
            model_name = model["name"]
            if model_name not in results:
                # first iteration
                results[model_name] = []

            gnn_model = None
            if model["name"] == "gcn_ff":
                print("Executing GCN + Feed Forward Network")
                gnn_model = GCN_FFN(
                    n_node_features=70,
                    h1=32,
                    n_gcn_output_features=32,
                    h2=32,
                    n_classes=2)
            elif model["name"] == "gat_ff":
                print("Executing GAT + Feed Forward Network")
                gnn_model = GAT_FFN(
                    n_node_features=70,
                    h1=32,
                    n_gcn_output_features=32,
                    h2=32,
                    n_classes=2)
            else:
                continue
            gnn_model = gnn_model.to(utils.get_device())
            result = run_gnn_model(gnn_model, train_data_loader, test_data_loader)
            result["model"]: model_name
            result["itr"]: itr
            results[model_name].append(result)

    # 5. Write the classification output
    utils.write_output(results, output_dir, output_prefix, "output")


def run_gnn_model(gnn_model, train_data_loader, test_data_loader):
    lr = 1e-4

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=5e-4)
    # CrossEntropyLoss applies log softmax to the output followed by computation of negative log likelihood
    # So, we did not include F.log_softmax() activation in the Feed Forward Network
    criterion = nn.CrossEntropyLoss()
    # Training
    gnn_model.train()
    for itr, batch in enumerate(pbar := tqdm.tqdm(train_data_loader)):
        ligand_graph, receptor_graph, pairs, labels = batch

        optimizer.zero_grad()

        output = gnn_model(ligand_graph, receptor_graph, pairs).to(utils.get_device()).squeeze()
        loss = criterion(output, labels.squeeze())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Classification/training-loss={float(loss.item())}, n_graph_pairs_procesed={itr+1}")

    gnn_model.eval()
    results = []
    for itr, batch in enumerate(pbar:= tqdm.tqdm(test_data_loader)):
        ligand_graph, receptor_graph, pairs, labels = batch

        output = gnn_model(ligand_graph, receptor_graph, pairs).to(utils.get_device()).squeeze()
        loss = criterion(output, labels.squeeze())
        pbar.set_description(f"Classification/validation-loss={float(loss.item())}, n_graph_pairs_procesed={itr+1}")
        # Explicity apply softmax to the output to get the probabilities, since
        # we did not include F.log_softmax() activation in the Feed Forward Network
        output = F.log_softmax(output, dim=-1)
        output = torch.argmax(output, dim=1, keepdim=True)
        results.append(pd.DataFrame({"y_pred": output.squeeze().numpy(), "y_true": labels.squeeze().numpy()}))
    return pd.concat(results, ignore_index=True)
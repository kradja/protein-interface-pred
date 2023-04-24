import os
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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
    pos_neg_ratio = classification_settings["pos_neg_ratio"]

    # 1. Read the training dataset files
    training_dataset = PDB_DB_Dataset(dirpath=os.path.join(input_dir, "train"), pos_neg_ratio=pos_neg_ratio)
    testing_dataset = PDB_DB_Dataset(dirpath=os.path.join(input_dir, "test"), pos_neg_ratio=pos_neg_ratio)

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
                print(f"Iteration {itr}: Executing GCN + Feed Forward Network")
                gnn_model = GCN_FFN(
                    n_node_features=70,
                    h1=32,
                    n_gcn_output_features=32,
                    h2=32,
                    n_classes=2)
            elif model["name"] == "gat_ff":
                print(f"Iteration {itr}: Executing GAT + Feed Forward Network")
                gnn_model = GAT_FFN(
                    n_node_features=70,
                    h1=32,
                    n_gcn_output_features=32,
                    h2=32,
                    n_classes=2)
            else:
                continue
            gnn_model = gnn_model.to(utils.get_device())
            result = run_gnn_model(gnn_model, train_data_loader, test_data_loader, model_name, classification_settings["n_epochs"])
            result["model"] = model_name
            result["itr"] = itr
            results[model_name].append(result)

    # 5. Write the classification output
    utils.write_output(results, output_dir, output_prefix, "output")


def run_gnn_model(gnn_model, train_data_loader, test_data_loader, model_name, n_epochs):
    tbw = SummaryWriter()

    lr = 1e-4
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-4,
        epochs=n_epochs,
        steps_per_epoch=len(train_data_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)
    # CrossEntropyLoss applies log softmax to the output followed by computation of negative log likelihood
    # So, we did not include F.log_softmax() activation in the Feed Forward Network
    criterion = nn.CrossEntropyLoss()
    # Training

    train_itr = 0
    test_itr = 0
    for epoch in range(n_epochs):
        gnn_model, train_itr = train_gnn_model(gnn_model, train_data_loader, optimizer, lr_scheduler, criterion, tbw, model_name, train_itr, epoch)
        _, test_itr = test_gnn_model(gnn_model, test_data_loader, criterion, tbw, model_name, test_itr, epoch, log_loss=True)
    results, _ = test_gnn_model(gnn_model, test_data_loader, criterion, tbw, model_name, itr=None, epoch=None, log_loss=False)
    return results


def train_gnn_model(gnn_model, train_data_loader, optimizer, lr_scheduler, criterion, tbw, model_name, itr, epoch):
    gnn_model.train()
    for _, batch in enumerate(pbar := tqdm.tqdm(train_data_loader)):
        itr += 1
        ligand_graph, receptor_graph, pairs, labels = batch

        optimizer.zero_grad()

        output = gnn_model(ligand_graph, receptor_graph, pairs).to(utils.get_device()).squeeze()
        loss = criterion(output, labels.squeeze())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        train_loss = float(loss.item())
        pbar.set_description(f"{model_name}/training-loss={train_loss}, epoch={epoch+1}")
        tbw.add_scalar(f"{model_name}/training-loss", train_loss, itr)
        tbw.add_scalar(f"{model_name}/learning-rate", float(lr_scheduler.get_last_lr()[0]), itr)
    return gnn_model, itr


def test_gnn_model(gnn_model, test_data_loader, criterion, tbw, model_name, itr, epoch, log_loss=True):
    gnn_model.eval()
    results = []
    for _, batch in enumerate(pbar := tqdm.tqdm(test_data_loader)):
        ligand_graph, receptor_graph, pairs, labels = batch

        output = gnn_model(ligand_graph, receptor_graph, pairs).to(utils.get_device()).squeeze()
        labels = labels.squeeze()
        pos_idx = (labels == 1).nonzero().squeeze()
        neg_idx = (labels == 0).nonzero().squeeze()

        pos_loss = criterion(torch.index_select(output, dim=0, index=pos_idx),
                             torch.index_select(labels, dim=0, index=pos_idx))
        neg_loss = criterion(torch.index_select(output, dim=0, index=neg_idx),
                             torch.index_select(labels, dim=0, index=neg_idx))

        val_loss = criterion(output, labels)
        pos_loss = float(pos_loss.item())
        neg_loss = float(neg_loss.item())

        if log_loss:
            itr += 1
            pbar.set_description(f"{model_name}/validation-loss={float(val_loss.item())}, epoch={epoch+1}")
            tbw.add_scalars(f"{model_name}/validation-loss", {"pos_loss": pos_loss, "neg_loss": neg_loss}, itr)
        # Explicity apply softmax to the output to get the probabilities, since
        # we did not include F.softmax() activation in the Feed Forward Network
        output = F.softmax(output, dim=-1)
        results.append(pd.DataFrame({
            "y_pred": output.squeeze().cpu().detach()[:, 1].numpy(),
            "y_true": labels.squeeze().cpu().numpy()}))
    return pd.concat(results, ignore_index=True), itr
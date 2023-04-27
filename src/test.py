import torch
import pickle
import pdb
import json
import os
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def focal_loss(bce_loss, label, alpha=0.9, gamma = 2):
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

class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x_ligand = F.relu(self.conv1(data.x_s, data.edge_index_s))
        x_ligand = F.relu(self.conv2(x_ligand, data.edge_index_s))

        x_receptor = F.relu(self.conv1(data.x_t, data.edge_index_t))
        x_receptor = F.relu(self.conv2(x_receptor, data.edge_index_t))

        x = torch.cat((x_ligand[data.y[:, 0]], x_receptor[data.y[:, 1]]), dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":# or key == "edge_attr_s":
            return self.x_s.size(0)
        if key == "edge_index_t":# or key == "edge_attr_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def _make_data_loader(data, batch_size, shuffle):
    """If I remove edge attributes my data loader works fine and can have multiple batches"""
    data_list = []
    for comp in data.values():
        l1 = comp["ligand"]
        r1 = comp["receptor"]
        lig_rec_data = PairData(
            x_s=l1["x"],
            edge_index_s=l1["edge_index"],
            edge_attr_s=l1["edge_attr"],
            x_t=r1["x"],
            edge_index_t=r1["edge_index"],
            edge_attr_t=r1["edge_attr"],
            y=comp["label"],
        )
        data_list.append(lig_rec_data)
    rr = DataLoader(
        data_list, batch_size, shuffle
    )  # , follow_batch=["x_s","x_t"])#collate_fn=collate_pair_data)
    return rr

def test_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_scores = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            pred = model(data)
            labels = data.y[:, 2].to(torch.float32)#.unsqueeze(1)
            y_true.append(labels)
            y_pred.append(pred.squeeze())
            y_scores.append(torch.tensor([1 if score >= 0.5 else 0 for score in y_pred[-1]]))
            bceloss = criterion(pred.squeeze().to(torch.float32), labels)
            loss = focal_loss(bceloss, labels)
            val_loss += loss.item()
    return y_true,y_scores,y_pred, val_loss

def main(train_file, test_file, output_file):
    data_train = torch.load(train_file)
    data_test = torch.load(test_file)
    train_loader = _make_data_loader(
        data_train, batch_size=1, shuffle=True
    )
    test_loader = _make_data_loader(
        data_test, batch_size=1, shuffle=True
    )
    
    input_dim = 70
    hidden_dim = 32
    #lr = 0.005
    epochs = 10 #number of training epochs

    # model, optimizer and loss function
    model = GCNClassifier(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-2,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0)

    # Iterations
    final_output = {"test": {}, "train": {}}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            pred = model(data)

            labels = data.y[:, 2].to(torch.float32)#.unsqueeze(1)
            bceloss = criterion(pred.squeeze().to(torch.float32), labels)
            loss = focal_loss(bceloss, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss += loss.item()

        y_true,y_scores,y_pred, val_loss= test_model(model, test_loader, criterion)
        y_true_train,y_scores_train,y_pred_train, loss_train= test_model(model, train_loader, criterion)

        print(f"Epoch {epoch + 1}/{epochs}, train_loss = {train_loss / len(train_loader):.4f}, val_loss = {val_loss / len(test_loader):.4f}")
        final_output["test"][f"Epoch {epoch + 1}"] = {"y_true": y_true, "y_scores": y_scores, "y_pred": y_pred}
        final_output["train"][f"Epoch {epoch + 1}"] = {"y_true": y_true_train, "y_scores": y_scores_train, "y_pred": y_pred_train}
        with open(output_file,'wb') as f:
            pickle.dump(final_output, f)

        
def eval_results(data):
    for epoch, results in data.items():
        print(f"Epoch {epoch}")
        y_true = torch.cat(results["y_true"], dim=0)
        y_scores = torch.cat(results["y_scores"], dim=0)
        y_pred = torch.cat(results["y_pred"], dim=0)
        # rr = np.count_nonzero(results["y_true"][0])/len(results["y_true"][0])
        evaluate(y_true, y_scores, y_pred)

def evaluate(y_true, y_scores, y_pred):
    accuracy = accuracy_score(y_true, y_scores)
    f1 = f1_score(y_true, y_scores)
    aucroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    print(f"Accuracy = {accuracy}, F1 = {f1}, AUCROC = {aucroc}, AUPRC = {auprc}")

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def input():
    params = {"raw": "../data/00_raw_data/", "processed": "../data/01_processed/",
              "model": "../data/02_models/"}
    complex_train_files = read_json(
        os.path.join(params["processed"], "complex_train.json")
    )
    complex_test_files = read_json(
        os.path.join(params["processed"], "complex_test.json")
    )
    output_predictions = os.path.join(params["processed"],'train_test_pred_program2.pkl')
    main(complex_train_files, complex_test_files, output_predictions)
    with open(output_predictions,'rb') as f:
        pred = pickle.load(f)
    print("________TEST________")
    eval_results(pred["test"])
    print("________TRAIN________")
    eval_results(pred["train"])

if __name__ == "__main__":
    input()
import torch
from torch_geometric.datasets import Planetoid
from src.prediction.models.gnn.gcn_2l import GCN_2Layer
import torch.nn.functional as F


def train_node_classification_model(device, dataset):
    n_epochs = 10
    lr = 0.01

    data = dataset[0].to(device)
    model = GCN_2Layer(n_node_features=dataset.num_node_features,
                       n_h=16,
                       n_classes=dataset.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model


def evaluate_node_classification_model(device, model, dataset):
    model.eval()
    data = dataset[0].to(device)
    predictions = model(data).argmax(dim=1)
    correct_predictions = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct_predictions)/int(data.test_mask.sum())
    print(f"Accuracy={accuracy:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    model = train_node_classification_model(device, dataset)
    evaluate_node_classification_model(device, model, dataset)


if __name__ == '__main__':
    main()

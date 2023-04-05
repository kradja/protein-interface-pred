import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.prediction.models.gnn.ffn import FFN_2L


class GCN_FFN(torch.nn.ModuleList):
    def __init__(self, n_node_features, h1, n_gcn_output_features, h2, n_classes):
        super(GCN_FFN, self).__init__()
        self.ligand_gcn = GCN_2L(n_node_features, h1, n_gcn_output_features)
        self.receptor_gcn = GCN_2L(n_node_features, h1, n_gcn_output_features)
        self.ffn = FFN_2L(2*n_gcn_output_features, h2, n_classes)

    def forward(self, data_l, data_r, X):
        x_ligand = self.ligand_gcn(data_l)
        x_receptor = self.ligand_gcn(data_r)
        # shape pf X is b x n x 2
        # since in our case, b is always 1: we are processing one pair of graphs at a time
        # we can squeeze X
        X = X.squeeze()
        l_nodes = x_ligand[X[:, 0], :]
        r_nodes = x_receptor[X[:, 1], :]
        pairs = torch.concat((l_nodes, r_nodes), dim=1)
        # unsqueeze it back
        pairs = pairs.unsqueeze(0)
        return self.ffn(pairs)


class GCN_2L(torch.nn.Module):
    def __init__(self, n_node_features, h, n_output_features):
        super().__init__()
        self.gcn_l1 = GCNConv(n_node_features, h)
        self.gcn_l2 = GCNConv(h, n_output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn_l1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_l2(x, edge_index)
        x = F.relu(x)
        return x


import torch.nn as nn
from torch_geometric.nn.conv import NNConv
import copy
import torch
from src.prediction.models.gnn.ffn import FFN_2L


class NNConv_FFN(nn.ModuleList):
    def __init__(self, n_node_features, n_edge_features, n_gnn_output_features, ff_h, n_classes):
        super(NNConv_FFN, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_edge_features, n_node_features),
                                 nn.ReLU(),
                                 nn.Linear(n_node_features, n_node_features * n_gnn_output_features))
        self.ligand_nnconv = NNConv(in_channels=n_node_features,
                                    out_channels=n_gnn_output_features,
                                    nn=copy.deepcopy(self.mlp))

        self.receptor_nnconv = NNConv(in_channels=n_node_features,
                                    out_channels=n_gnn_output_features,
                                    nn=copy.deepcopy(self.mlp))
        self.ffn = FFN_2L(2 * n_gnn_output_features, ff_h, n_classes)

    def forward(self, data_l, data_r, X):
        x_ligand = self.ligand_nnconv(data_l.x, data_l.edge_index, data_l.edge_attr.type(torch.float32))
        x_receptor = self.receptor_nnconv(data_r.x, data_r.edge_index, data_r.edge_attr.type(torch.float32))
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

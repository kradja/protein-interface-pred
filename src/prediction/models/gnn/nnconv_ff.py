import torch
import copy
import torch.nn as nn
from torch_geometric.nn.conv import NNConv
import torch.nn.functional as F
from src.prediction.models.gnn.ffn import FFN_2L


class NNConv_FFN(nn.ModuleList):
    def __init__(self, n_node_features, n_edge_features, n_gnn_output_features, nnconv_h, ff_h, n_classes):
        super(NNConv_FFN, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(n_edge_features, nnconv_h),
                                  nn.ReLU(),
                                  nn.Linear(nnconv_h, n_node_features * n_gnn_output_features))
        self.ligand_nnconv = NNConv(in_channels=n_node_features,
                                out_channels=n_gnn_output_features,
                                nn=copy.deepcopy(self.mlp))
        self.receptor_nnconv  = NNConv(in_channels=n_node_features,
                                    out_channels=n_gnn_output_features,
                                    nn=copy.deepcopy(self.mlp))
        # self.ligand_nnconv = NNConv_2L(n_node_features=n_node_features,
        #                                n_edge_features=n_edge_features,
        #                                h1=nnconv_h1,
        #                                h2=nnconv_h2,
        #                                n_gnn_output_features=n_gnn_output_features)
        #
        # self.receptor_nnconv = NNConv_2L(n_node_features=n_node_features,
        #                                  n_edge_features=n_edge_features,
        #                                  h1=nnconv_h1,
        #                                  h2=nnconv_h2,
        #                                  n_gnn_output_features=n_gnn_output_features)
        self.ffn = FFN_2L(2 * n_gnn_output_features, ff_h, n_classes)

    def forward(self, data_l, data_r, X):
        x_ligand = self.ligand_nnconv(data_l.x, data_l.edge_index, data_l.edge_attr.type(torch.float32))
        x_receptor = self.receptor_nnconv(data_r.x, data_r.edge_index, data_r.edge_attr.type(torch.float32))
        # x_ligand = self.ligand_nnconv(data_l)
        # x_receptor = self.receptor_nnconv(data_r)
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


class NNConv_2L(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, h1, h2, n_gnn_output_features):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(n_edge_features, h1),
                                  nn.ReLU(),
                                  nn.Linear(h1, n_node_features * h2))
        self.nnconv_l1 = NNConv(in_channels=n_node_features,
                                out_channels=h2,
                                nn=self.mlp1)

        self.mlp2 = nn.Sequential(nn.Linear(n_edge_features, h1),
                                  nn.ReLU(),
                                  nn.Linear(h1, h2 * n_gnn_output_features))
        self.nnconv_l2 = NNConv(in_channels=h2,
                                out_channels=n_gnn_output_features,
                                nn=self.mlp2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.type(torch.float32)
        x = self.nnconv_l1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv_l2(x, edge_index, edge_attr)
        x = F.relu(x)
        return x

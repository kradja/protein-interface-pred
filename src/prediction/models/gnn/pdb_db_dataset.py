import pandas as pd
import torch
import torch_geometric.utils
from torch_geometric.data import Dataset
from os import path
import networkx as nx

from src.utils import utils


class PDB_DB_Dataset(Dataset):
    def __init__(self, dirpath, pos_neg_ratio):
        super(PDB_DB_Dataset, self).__init__()
        self.dirpath = dirpath
        self.pos_neg_ratio = pos_neg_ratio
        self.prot_complex_ids = None
        self.n_prot_complex_ids = None
        self.prot_interface_pairs = None
        self.init_references(dirpath)

    def init_references(self, dirpath):
        # Read the protein ids
        prot_complex_ids_filepath = path.join(dirpath, "protein_complex_ids.txt")
        print(f"Reading protein complex ids from {prot_complex_ids_filepath}")
        with open(prot_complex_ids_filepath, "r") as f:
            self.prot_complex_ids = f.read().split("\n")
        self.n_prot_complex_ids = len(self.prot_complex_ids)

        # Read the protein interface protein pairs
        prot_interface_labels_filepath = path.join(dirpath, "protein_interface_labels.csv")
        print(f"Reading protein interface labels from {prot_interface_labels_filepath}")
        df = pd.read_csv(prot_interface_labels_filepath)
        if self.pos_neg_ratio:
            pos_df = df[df["label"] == 1]
            neg_df = df[df["label"] == 0]
            n_pos = pos_df.shape[0]
            n_neg = int(n_pos/self.pos_neg_ratio)
            neg_df = neg_df.head(n_neg)
            df = pd.concat([pos_df, neg_df])
        self.prot_interface_pairs = df

    def len(self):
        return self.n_prot_complex_ids

    def get(self, idx: int):
        prot_complex_id = self.prot_complex_ids[idx]
        ligand_graph_nx = nx.read_gpickle(path.join(self.dirpath, f"{idx}_{prot_complex_id}_l.gpickle"))
        # from_networkx() does not store all the attributes of the node by default
        # hack
        # 1: get list of all attribute keys from the networkx graph and pass it as an argument to from_networkx() to load them
        # 2: attribute keys in the original networkx graph are integers 0, 1, 2, ....
        #          however, torch requires the keys to be strings, so we convert them
        # 3: assume the ligand and receptor have the same attribute keys
        node_attrs = [str(x) for x in list(ligand_graph_nx.nodes[0].keys())]

        ligand_graph = torch_geometric.utils.from_networkx(ligand_graph_nx, group_node_attrs=node_attrs)
        receptor_graph = torch_geometric.utils.from_networkx(
            nx.read_gpickle(
                path.join(self.dirpath, f"{idx}_{prot_complex_id}_r.gpickle")
            ), group_node_attrs=node_attrs
        )
        del ligand_graph_nx  # clear the object to save memory
        prot_complex_interface_pairs = self.prot_interface_pairs[self.prot_interface_pairs["complex_id"] == prot_complex_id]
        pairs = torch.tensor(prot_complex_interface_pairs[["l_index", "r_index"]].values, device=utils.get_device())
        labels = torch.tensor(prot_complex_interface_pairs[["label"]].values, device=utils.get_device())
        return ligand_graph.to(utils.get_device()), receptor_graph.to(utils.get_device()), pairs, labels
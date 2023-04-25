import os
import pdb
import pickle

import numpy as np
import torch
from torch_geometric.transforms import NormalizeScale



def convert_data(input_file):
    with open(input_file, "rb") as f:
        data = pickle.load(f, encoding="latin")
    return data


def write_tensors(data, processed_path, file_descrip):
    """Writing tensors for each complex to processed folder"""
    complex_data = dict()
    for ind, complex_id in enumerate(data[0]):
        # Features for ligand and receptor
        ligand_aa_features = data[1][ind]["l_vertex"]
        receptor_aa_features = data[1][ind]["r_vertex"]
        # length_ligand = len(ligand_aa_features)
        # length_receptor= len(receptor_aa_features)

        # Labels for pairs between ligand and receptor
        aa_label = data[1][ind]["label"]
        mask = aa_label[:, 2] == -1
        aa_label[mask, 2] = 0

        # Edge Attributes for ligand and Receptor
        ligand_edge_attr = data[1][ind]["l_edge"]
        ligand_edge_attr = ligand_edge_attr.reshape(
            np.multiply(*ligand_edge_attr.shape[:-1]), 2
        )

        receptor_edge_attr = data[1][ind]["r_edge"]
        receptor_edge_attr = receptor_edge_attr.reshape(
            np.multiply(*receptor_edge_attr.shape[:-1]), 2
        )
        pdb.set_trace()

        # Edge Indices for ligand and Receptor
        ligand_ind = data[1][ind]["l_hood_indices"]
        ligand_edges = ligand_ind.reshape(*ligand_ind.shape[:-1])
        ligand_edges_row = np.array(
            [
                np.repeat(x, ligand_edges.shape[1])
                for x in np.arange(ligand_edges.shape[0])
            ]
        )
        ligand_edge_index = np.vstack(
            (ligand_edges_row.flatten(), ligand_edges.flatten())
        )

        receptor_ind = data[1][ind]["r_hood_indices"]
        receptor_edges = receptor_ind.reshape(*receptor_ind.shape[:-1])
        receptor_edges_row = np.array(
            [
                np.repeat(x, receptor_edges.shape[1])
                for x in np.arange(receptor_edges.shape[0])
            ]
        )
        receptor_edge_index = np.vstack(
            (receptor_edges_row.flatten(), receptor_edges.flatten())
        )
        # ligand = Data(x=torch.tensor(ligand_aa_features, dtype=torch.float), edge_index=torch.tensor(ligand_edge_index, dtype=torch.long), edge_attr=torch.tensor(ligand_edge_attr, dtype=torch.float))
        # receptor = Data(x=torch.tensor(receptor_aa_features, dtype=torch.float), edge_index=torch.tensor(receptor_edge_index, dtype=torch.long), edge_attr=torch.tensor(receptor_edge_attr, dtype=torch.float))
        # data.x = (data.x - data.x.min()) / (data.x.max() - data.x.min())
        complex_tensors = {
            "ligand": {
                "x": torch.tensor(ligand_aa_features, dtype=torch.float),
                "edge_index": torch.tensor(ligand_edge_index, dtype=torch.long),
                "edge_attr": torch.tensor(ligand_edge_attr, dtype=torch.float),
            },
            "receptor": {
                "x": torch.tensor(receptor_aa_features, dtype=torch.float),
                "edge_index": torch.tensor(receptor_edge_index, dtype=torch.long),
                "edge_attr": torch.tensor(receptor_edge_attr, dtype=torch.float),
            },
            "label": torch.tensor(aa_label, dtype=torch.long),
        }
        # complex_tensor_file = os.path.join(processed_path,f"{complex_id}_tensors.pt")
        # torch.save(complex_tensors, complex_tensor_file)
        complex_data[complex_id] = complex_tensors  # complex_tensor_file

    complex_tensor_file = os.path.join(processed_path, f"{file_descrip}_tensors.pt")
    torch.save(complex_data, complex_tensor_file)
    return complex_tensor_file

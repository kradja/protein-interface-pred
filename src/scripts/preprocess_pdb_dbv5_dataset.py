import argparse
import os
import networkx as nx

import pandas as pd
import numpy as np

from pathlib import Path

from src.utils import utils


label_col = "label"
l_vertex_col = "l_vertex"
r_vertex_col = "r_vertex"


def parse_args():
    parser = argparse.ArgumentParser(description='Read and preprocess Protein Data Bank: Docking Benchamrk Dataset v5')
    parser.add_argument('-if', '--input_filepath', required=True,
                        help="Absolute path to the input dataset file\n")
    parser.add_argument('-od', '--output_dirpath', required=True,
                        help="Absolute path of the output directory\n")
    parser.add_argument('-t', '--type', required=True,
                        help="Supported values: features, graphs. Type of dataset to be constructed: node features or graphs\n")
    parser.add_argument('-m', '--max_complexes', required=False,
                        help="Optional: Maximum number of protein complexes to be processed\n")
    args = parser.parse_args()
    return args


def construct_graph(nodes, node_neighborhoods) -> nx.Graph:
    G = nx.Graph()
    n = len(nodes) # number of nodes in graph
    n_neighbors = 20 # number of neighbors for each node

    # nodes = numpy array of shape n x 70 where n is the number of nodes (residues)
    # and 70 is the number of features
    nodes_df = pd.DataFrame(nodes)
    # convert to row-wise maps. Example: [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]
    node_features = nodes_df.to_dict(orient="records")
    G.add_nodes_from(enumerate(node_features))
    # remove axis of length one
    # node neighborhoods size = n x 20 where n is the number of nodes (residues)
    # and 20 is for the 20 closest neighbor residues
    node_neighborhoods = np.squeeze(node_neighborhoods)
    for i in range(n):
        for j in range(n_neighbors):
            G.add_edge(i, j)
    return G


def construct_features_dataset(dataset, outputdir, filename, max_complexes):
    # construct the dataset as follows:
    #  - for each complex
    #    - for each element in labels attribute of the form (l_vertex_idx, r_vertex_idx, label)
    #       - get the features of l_vertex corresponding to l_vertex_idx
    #       - get the features of r_vertex corresponding to r_vertex_idx
    #       - concatenate the features
    #       - assign the label


    features_col = "amino_acid_pair_features"
    data_rows = []
    # dataset[0] is the list of protein complex ids
    # dataset[1] is the list of data for each protein complex
    prot_complexes = dataset[1]
    # hack to tackle insufficient memory
    if max_complexes:
        prot_complexes = prot_complexes[:int(max_complexes)]

    for prot_complex in prot_complexes:
        l_vertex_features = prot_complex[l_vertex_col]
        r_vertex_features = prot_complex[r_vertex_col]

        amino_acid_pairs = prot_complex[label_col]
        for amino_acid_pair in amino_acid_pairs:
            l_vertex_idx = amino_acid_pair[0]
            r_vertex_idx = amino_acid_pair[1]
            label = amino_acid_pair[2]

            l_vertex = l_vertex_features[l_vertex_idx]
            r_vertex = r_vertex_features[r_vertex_idx]
            data_rows.append((np.concatenate((l_vertex, r_vertex)), label))
    df = pd.DataFrame(data_rows, columns=[features_col, label_col])
    print(f"Size of data_rows = {len(data_rows)}")
    print(f"Size of dataset = {df.shape}")

    # split the amino_acid_pair_features col into individual columns
    df = pd.concat([df[label_col], pd.DataFrame(df[features_col].to_list())], axis=1)
    # replace all -1 labels with 0 for downstream compatibility
    df.loc[df[label_col] == -1, label_col] = 0
    print(f"Size of dataset = {df.shape}")
    df.to_csv(os.path.join(outputdir, f"{filename}.csv"), index=False)


def construct_graphs(dataset, output_dir):
    # create any missing parent directories
    Path(os.path.dirname(output_dir)).mkdir(parents=True, exist_ok=True)

    prot_complexes = dataset[1]
    print(f"Number of protein complexes = {len(prot_complexes)}")

    # for each protein complex
    for i, prot_complex in enumerate(prot_complexes):
        complex_id = prot_complex["complex_code"]

        G_l = construct_graph(prot_complex[l_vertex_col], prot_complex["l_hood_indices"])
        print(f"Constructed Complex {i}:{complex_id}-Ligand --> "
              f"nodes={G_l.number_of_nodes()}, edges={G_l.number_of_edges()}")

        G_r = construct_graph(prot_complex[r_vertex_col], prot_complex["r_hood_indices"])
        print(f"Constructed Complex {i}:{complex_id}-Receptor --> "
              f"nodes={G_r.number_of_nodes()}, edges={G_r.number_of_edges()}")

        nx.write_gpickle(G_l, os.path.join(output_dir, f"{i}_{complex_id}_l.gpickle"))
        nx.write_gpickle(G_r, os.path.join(output_dir, f"{i}_{complex_id}_r.gpickle"))


def write_metadata(dataset, output_dirpath):
    # Write the list of protein complex ids in the dataset
    prot_complex_ids = dataset[0]
    output_filepath = os.path.join(output_dirpath,  "protein_complex_ids.txt")
    print(f"Writing protein complex ids to {output_filepath}")
    with open(output_filepath, "w+") as f:
        f.write("\n".join(prot_complex_ids))

    # For each protein complex, write the residue pairs with labels for interface prediction
    labeled_interface_pairs = []
    for prot_complex in dataset[1]:
        complex_id = prot_complex['complex_code']
        amino_acid_pairs = prot_complex[label_col]
        df = pd.DataFrame(amino_acid_pairs, columns=["l_index", "r_index", label_col])
        df["complex_id"] = complex_id
        # replace all -1 labels with 0 for compatibility with GNN models
        df.loc[df[label_col] == -1, label_col] = 0
        labeled_interface_pairs.append(df)

    output_filepath = os.path.join(output_dirpath, "protein_interface_labels.csv")
    df = pd.concat(labeled_interface_pairs, ignore_index=True)
    df.to_csv(output_filepath, index=False)


def main():
    args = parse_args()
    input_filepath = args.input_filepath
    output_dirpath = args.output_dirpath

    print(f"Reading dataset from {input_filepath}")
    dataset = utils.read_dataset(input_filepath)

    if args.type == "features":
        filename = os.path.basename(os.path.realpath(input_filepath))
        construct_features_dataset(dataset, output_dirpath, filename, args.max_complexes)
    elif args.type == "graphs":
        construct_graphs(dataset, output_dirpath)
        write_metadata(dataset, output_dirpath)

    else:
        print(f"ERROR: Dataset type {args.type} is not supported")


if __name__ == '__main__':
    main()

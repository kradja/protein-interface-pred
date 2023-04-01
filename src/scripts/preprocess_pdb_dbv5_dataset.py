import argparse
import os
import networkx as nx
import pandas as pd
import numpy as np

from pathlib import Path

from src.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Read and preprocess Protein Data Bank: Docking Benchamrk Dataset v5')
    parser.add_argument('-if', '--input_filepath', required=True,
                        help="Absolute path to the input dataset file\n")
    parser.add_argument('-od', '--output_dirpath', required=True,
                        help="Absolute path of the output directory\n")
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


def construct_graphs(dataset, output_dir):
    # create any missing parent directories
    Path(os.path.dirname(output_dir)).mkdir(parents=True, exist_ok=True)

    prot_complexes = dataset[1]
    print(f"Number of protein complexes = {len(prot_complexes)}")

    # for each protein complex
    for i, prot_complex in enumerate(prot_complexes):
        complex_code = prot_complex['complex_code']

        G_l = construct_graph(prot_complex["l_vertex"], prot_complex["l_hood_indices"])
        print(f"Constructed Complex {i}:{complex_code}-Ligand --> "
              f"nodes={G_l.number_of_nodes()}, edges={G_l.number_of_edges()}")

        G_r = construct_graph(prot_complex["r_vertex"], prot_complex["r_hood_indices"])
        print(f"Constructed Complex {i}:{complex_code}-Receptor --> "
              f"nodes={G_r.number_of_nodes()}, edges={G_r.number_of_edges()}")

        nx.write_adjlist(G_l, os.path.join(output_dir, f"{i}_{complex_code}_l.adjlist"))
        nx.write_adjlist(G_r, os.path.join(output_dir, f"{i}_{complex_code}_r.adjlist"))


def main():
    args = parse_args()
    input_filepath = args.input_filepath
    output_dirpath = args.output_dirpath

    print(f"Reading dataset from {input_filepath}")
    dataset = utils.read_dataset(input_filepath)
    construct_graphs(dataset, output_dirpath)


if __name__ == '__main__':
    main()

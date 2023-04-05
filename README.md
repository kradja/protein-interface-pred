# Protein Interface Prediction
Protein Interface Prediction utilizing Graph ML


### Scripts
#### Preprocessing Protein Data Bank: Docking Benchamrk Dataset v5
`src/scripts/preprocess_pdb_dbv5_dataset.py`

**Usage**: 
`python src/scripts/preprocess_pdb_dbv5_dataset.py -if <absolute path to pickle file> -od <absolute path to output directory>`

Example: `python src/scripts/preprocess_pdb_dbv5_dataset.py -if C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\train.cpkl -od C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\protein_complex_graphs`


This script will - 
1. Read the dataset file
2. Create individual networkx graphs for ligands and receptors of each protein complex
3. Write each networkx graph in 'gpickle' format in the specified output directory
4. Write a file with list of protein complex ids
5. Write a file containing indices of residue pairs with labels for interface prediction
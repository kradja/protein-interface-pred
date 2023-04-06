# Protein Interface Prediction
Protein Interface Prediction utilizing Graph ML


## Dataset Preprocessing
### Preprocessing Protein Data Bank: Docking Benchamrk Dataset v5
`src/scripts/preprocess_pdb_dbv5_dataset.py`

**Usage**: 
`python -m src.scripts.preprocess_pdb_dbv5_dataset -if <absolute path to pickle file> -od <absolute path to output directory> -t <type>`

Example: 

Graphs
`python -m src.scripts.preprocess_pdb_dbv5_dataset -if C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\train.cpkl -od C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\protein_complex_graphs\train -t graphs`


Features
`python -m src.scripts.preprocess_pdb_dbv5_dataset -if C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\train.cpkl -od C:\Dev\git\protein-interface-pred\input\data\pdb-dbv5\residue_features_dataset -t features`



This script will -
- Graphs
    1. Read the dataset file
    2. Create individual networkx graphs for ligands and receptors of each protein complex
    3. Write each networkx graph in 'gpickle' format in the specified output directory
    4. Write a file with list of protein complex ids 
    5. Write a file containing indices of residue pairs with labels for interface prediction
- Features
    1. Read the dataset file
    2. For each protein complex
    3. Concatenate the feature vectors for all pairs of residue given in 'label' to generate features for each record
    3. Add the label (1 or 0) to this record 
    4. Write to csv file
    

## Prediction
There are two types of prediction supported by this pipeline - 
1. Baseline models
- Uses features dataset
- Models: Logistic Regression, Random Forest
- Config file: [pdb-dbv5-prot-interface-baseline-pred-yaml](/input/configs/prediction/pdb-dbv5-prot-interface-baseline-pred.yaml)
2. GNN models
- Uses graphs dataset
- Models: GCN + FFN, GAT + FFN
- Config file: [pdb-dbv5-prot-interface-gnn-pred-yaml](/input/configs/prediction/pdb-dbv5-prot-interface-gnn-pred.yaml)


**Usage**: 
`python -m src.protein_interface_prediction --config <absolute path to config file>`

Example: 
`python -m src.protein_interface_prediction --config C:\Dev\git\protein-interface-pred\input\configs\prediction\pdb-dbv5-prot-interface-gnn-pred.yaml`


## Evaluation
Generate and compare the performance metrics of models.
Metrics supported by the pipeline: **accuracy, f1, auroc, auprc**

Config file: [pdb-dbv5-prot-interface-eval.yaml](/input/configs/evaluation/pdb-dbv5-prot-interface-eval.yaml)


**Usage**: 
`python -m src.protein_interface_prediction --config <absolute path to config file>`

Example: 
`python -m src.protein_interface_prediction --config C:\Dev\git\protein-interface-pred\input\configs\evaluation/pdb-dbv5-prot-interface-eval.yaml`

Raw metrics are written to a csv file within output/evaluation, and visualiations are created within output/visualizations
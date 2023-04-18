import torch
import os
import json
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv 

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x,edge_index):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            # x = self.dropout(x)
        return x

class FFN_GCNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers ,dropout):
        super(FFN_GCNs, self).__init__()
        self.gcn = GCN(in_dim, hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim*2, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, edge_index1, x2, edge_index2,label):
        x_ligand = self.gcn(x1,edge_index1)
        x_receptor = self.gcn(x2,edge_index2)
        # First and second columns of label
        x = torch.cat((x_ligand[label[:,0]], x_receptor[label[:,1]]), dim=1)
        x = self.fc(x)
        x = F.relu(x)
        # x = self.dropout(x)
        return x

class PairData(Data):
    def __init__(self, x_s=None, edge_index_s=None, edge_attr_s=None, x_t=None, edge_index_t=None, edge_attr_t=None, y=None):
        super().__init__()
        self.x_s = x_s
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_t = x_t
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.y = y

    @property
    def num_nodes_s(self):
        return self.x_s.size(0) + self.x_t.size(0)

    @property
    def num_node_features(self):
        return self.x_s.size(1)

    def __inc__(self, key, value,*args,**kwargs):
        if key == 'edge_index_s' or key == 'edge_attr_s':
            return self.x_s.size(0)
        if key == 'edge_index_t' or key == 'edge_attr_t':
            return self.x_t.size(0)
        return super().__inc__(key, value,*args,**kwargs)

#class PairDataset(DataLoader):
#    def __init__(self, data_list):
#        self.data_list = data_list
#
#    def __len__(self):
#        return len(self.data_list)
#
#    def __getitem__(self, index):
#        return self.data_list[index]

def _make_data_loader(data, batch_size, shuffle):
    """If I remove edge attributes my data loader works fine and can have multiple batches"""
    data_list = []
    for comp in data.values():
        l1 = comp['ligand']
        r1 = comp['receptor']
        lig_rec_data = PairData(x_s=l1['x'],edge_index_s=l1['edge_index'], edge_attr_s=l1['edge_index'],
                                x_t=r1['x'],edge_index_t=r1['edge_index'],edge_attr_t=r1['edge_index'],y=comp['label'])
        data_list.append(lig_rec_data)
    rr = DataLoader(data_list, batch_size, shuffle)#, follow_batch=["x_s","x_t"])#collate_fn=collate_pair_data)
    return rr

def _train(model, crit, optimizer, input_data):
    model.train()
    for batch in input_data:
        optimizer.zero_grad()
        # y = batch.y[:,2]
        output = model(batch.x_s, batch.edge_index_s, batch.x_t, batch.edge_index_t,batch.y)
        pdb.set_trace()
        loss = crit(output, batch.y[:,2])
        loss.backward()
        optimizer.step()
    return loss.item()

#def collate_pair_data(batch):
#    pdb.set_trace()
#    x_list, edge_index_list, edge_attr_list, y_list = [], [], [], []
#    for data in batch:
#        print(data)
#        x_list.append(data.x)
#        edge_index_list.append(data.edge_index)
#        edge_attr_list.append(data.edge_attr)
#        y_list.append(data.y)
#    return PairData(
#        x=torch.cat(x_list, dim=0),
#        edge_index=torch.cat(edge_index_list, dim=1),
#        edge_attr=torch.cat(edge_attr_list, dim=0),
#        y=torch.tensor(y_list)
#    )

def run_gcn(train,test):
    results = {}
    data_train = torch.load(train)
    data_test = torch.load(test)
    # Batch_size should be larger!
    train_loader = _make_data_loader(data_train, batch_size=1, shuffle=True)#,num_workers=2)
    test_loader = _make_data_loader(data_test, batch_size=1, shuffle=False) #,num_workers=2)

    # Run GCN (The number of features is always 70)
    model = FFN_GCNs(in_dim=70, hidden_dim=32, out_dim=2, num_layers=2, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    loss = _train(model, crit, optimizer, train_loader)
    #loss = train(model, crit, optimizer, data, label)
    #print("Epoch: {}, Loss: {}".format(epoch, loss))
    print("Done")

#def train(model, crit, optimizer, data, label):
#    optimizer.zero_grad()
#    output = model(data)
#    loss = crit(output, label)
#    loss.backward()
#    optimizer.step()
#    return loss.item()
#
#def test(model, crit, data, label):
#    output = model(data)
#    loss = crit(output, label)
#    return loss.item()
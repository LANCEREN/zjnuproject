import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, concat=True, dropout=dropout)
        self.conv2 = GATConv(nhid * nheads, nhid, heads=nheads, concat=True, dropout=dropout)
        self.conv3 = GATConv(nhid * nheads, nclass, heads=nheads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Conv1 + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)  # Dropout after conv1

        # Conv2 + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)  # Dropout after conv2

        # Conv3 (no activation, just dropout before classification)
        embeddings = x
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1), embeddings

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__(c)
        self.dropout = dropout
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nhid)
        self.conv3 = SAGEConv(nhid, nclass)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        embeddings = x
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings




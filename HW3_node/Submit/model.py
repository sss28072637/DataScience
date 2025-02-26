import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class YourGNNModel(nn.Module):
    """
    Your Graph Neural Network Model:
    - A simple GNN model for node classification
    """
    def __init__(self, in_size, hid_size, out_size):
        super(YourGNNModel, self).__init__()
        self.conv1 = GraphConv(in_size, hid_size)
        self.conv2 = GraphConv(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        h = F.relu(self.conv1(g, h))
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h
    
    
class CRD(nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class Net(nn.Module):
    def __init__(self, features, num_classes, hidden_size, dropout):
        super(Net, self).__init__()
        self.crd = CRD(features.size(1), hidden_size, dropout)
        self.cls = CLS(hidden_size, num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, graph, features, train_mask):
        x, edge_index = features, graph.edge_index
        x = self.crd(x, edge_index, train_mask)
        x = self.cls(x, edge_index, train_mask)
        return x

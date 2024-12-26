import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from models.kan.KANLayer import *

class GNN_KAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kan_hidden_dim):
        super(GNN_KAN, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.kan = KANLayer(hidden_dim, kan_hidden_dim)
        self.classifier = nn.Linear(kan_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = self.kan(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

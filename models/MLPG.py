import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GNN_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_hidden_dim):
        super(GNN_MLP, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(mlp_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
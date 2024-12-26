import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from train import *

dataset = Planetoid(root="./data", name="PubMed")
data = dataset[0]

input_dim = dataset.num_node_features
hidden_dim = 32
output_dim = dataset.num_classes
hidden_dim2 = 32


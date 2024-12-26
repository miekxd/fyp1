import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Reddit2, Amazon, Twitch, PPI, QM9, ZINC
from ogb.nodeproppred import PygNodePropPredDataset
from models.KANG import *
from models.MLPG import *

def load_dataset(dataset_name):
    if dataset_name == "PubMed":
        return Planetoid(root="./data", name="PubMed")
    elif dataset_name == "Cora":
        return Planetoid(root="./data", name="Cora")
    elif dataset_name == "CiteSeer":
        return Planetoid(root="./data", name="CiteSeer")
    elif dataset_name == "Reddit":
        return Reddit2(root="./data")
    elif dataset_name == "Amazon-Computers":
        return Amazon(root="./data", name="Computers")
    elif dataset_name == "Amazon-Photo":
        return Amazon(root="./data", name="Photo")
    elif dataset_name == "Twitch-EN":
        return Twitch(root="./data", name="EN")
    elif dataset_name == "PPI":
        return PPI(root="./data")
    elif dataset_name == "ogbn-arxiv":
        return PygNodePropPredDataset(name="ogbn-arxiv")
    elif dataset_name == "ogbn-products":
        return PygNodePropPredDataset(name="ogbn-products")
    else:
        raise ValueError("Dataset not recognized")
    
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []

    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

dataset_name='Reddit'
mode='KAN'

dataset = load_dataset(dataset_name)
print(f"Loaded dataset: {dataset_name}")
data=dataset[0]

input_dim = dataset.num_node_features
hidden_dim = 32
output_dim = dataset.num_classes
hidden_dim2 = 32

if mode=='MLP':
    model=GNN_MLP(input_dim, hidden_dim, output_dim, hidden_dim2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
elif mode == 'KAN':
    model=GNN_KAN(input_dim, hidden_dim, output_dim, hidden_dim2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")




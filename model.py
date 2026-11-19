"""A compact two-layer edge-aware Graph Attention Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AMLGAT(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=16, dropout=0.25):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(
            in_channels, hidden_channels, heads=2, concat=True,
            edge_dim=edge_dim, dropout=dropout
        )
        self.gat2 = GATConv(
            hidden_channels * 2, hidden_channels, heads=1, concat=False,
            edge_dim=edge_dim, dropout=dropout
        )
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        return self.classifier(x).squeeze(-1)  # raw logits

"""Two-layer edge-aware GAT for transaction-level binary classification."""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TransactionGAT(nn.Module):
    def __init__(self, node_features=5, edge_features=2, hidden_channels=16, dropout=.25):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(node_features, hidden_channels, heads=2, concat=True,
                            edge_dim=edge_features, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False,
                            edge_dim=edge_features, dropout=dropout, add_self_loops=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_features, hidden_channels),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, context_edge_attr,
                query_src, query_dst, query_edge_attr):
        h = F.elu(self.gat1(x, edge_index, context_edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.gat2(h, edge_index, context_edge_attr))
        z = torch.cat([h[query_src], h[query_dst], query_edge_attr], dim=1)
        return self.classifier(z).squeeze(-1)

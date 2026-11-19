
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class AMLGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        heads=4,
        edge_dim=None,
        dropout=0.3
    ):
        super().__init__()

        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim
        )

        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        return self.classifier(x).squeeze(-1)


# Backward-compatible class name
GAT = AMLGAT

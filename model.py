"""Edge-aware Graph Attention Network for account-level AML classification.

Architecture
------------
    x (N, F_node) --dropout--> GATConv #1 (heads=H, concat) --ELU--> (N, hidden*H)
                  --dropout--> GATConv #2 (heads=1)         --ELU--> (N, hidden)
                  --> Linear --> logit (N, 1)

Both GATConv layers receive ``edge_attr`` (amount, time, currency, payment
format), so the attention score for a neighbour j of node i depends on:

    e_ij = LeakyReLU( a^T [ W h_i || W h_j || W_e edge_attr_ij ] )

The model returns RAW LOGITS. Do not apply a sigmoid here: train.py uses
BCEWithLogitsLoss (which applies it internally, in a numerically stable way)
and applies torch.sigmoid itself when computing metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class AMLGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention, head outputs are concatenated,
        # so the output width is hidden_channels * heads.
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            edge_dim=edge_dim, dropout=dropout,  # dropout on attention coefficients
        )
        # Layer 2: single head, one hidden vector per account.
        self.conv2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False,
            edge_dim=edge_dim, dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, return_attention: bool = False):
        """Return logits of shape (N, 1).

        If ``return_attention`` is True, also return the attention weights of
        both layers as [(edge_index_1, alpha_1), (edge_index_2, alpha_2)].
        Note GATConv appends self-loops, so edge_index here is longer than
        the input edge_index. Useful for interpreting which neighbours/
        transactions the model attends to.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, att1 = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, att2 = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        logits = self.classifier(x)
        if return_attention:
            return logits, [att1, att2]
        return logits

"""Edge-aware GNN with edge updates (GINE layers + an edge-update MLP per layer).

Per layer: GINE message passing (sum of neighbour messages enriched with edge embeddings) on a
bidirectional graph, then an edge update  e <- e + MLP([h_src, h_dst, e]).
The classifier reads [h_src, h_dst, e] for each transaction (edge classification).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class AMLGNN(nn.Module):
    def __init__(self, node_in, edge_in, h=64, layers=2, drop=0.1):
        super().__init__()
        self.node_enc, self.edge_enc = nn.Linear(node_in, h), mlp(edge_in, h, h)
        self.convs = nn.ModuleList(GINEConv(mlp(h, h, h), edge_dim=h) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(h) for _ in range(layers))
        self.upd = nn.ModuleList(mlp(3 * h, h, h) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
        self.drop = drop

    def forward(self, x, ei, ea):
        h, e = F.relu(self.node_enc(x)), self.edge_enc(ea)
        ei2 = torch.cat([ei, ei.flip(0)], 1)  # bidirectional message passing
        for conv, nm, upd in zip(self.convs, self.norms, self.upd):
            h = h + F.dropout(F.relu(nm(conv(h, ei2, torch.cat([e, e], 0)))), self.drop, self.training)
            e = e + upd(torch.cat([h[ei[0]], h[ei[1]], e], 1))
        return self.head(torch.cat([h[ei[0]], h[ei[1]], e], 1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(model, data, lo, hi, chunk, device):
    """Laundering probability for transactions [lo, hi), processed in time-contiguous chunks."""
    model.eval()
    out = []
    for a in range(lo, hi, chunk):
        x, ei, ea, _ = data.make_chunk(a, min(a + chunk, hi), device)
        out.append(torch.sigmoid(model(x, ei, ea)).cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Checkpoint helpers (the checkpoint stores the config so evaluate.py can rebuild everything)
# --------------------------------------------------------------------------- #
def save_checkpoint(path, state_dict, cfg: dict, node_in: int, edge_in: int, best_val_ap: float, best_epoch: int):
    torch.save({"state_dict": state_dict, "config": cfg, "node_in": node_in, "edge_in": edge_in,
                "best_val_pr_auc": float(best_val_ap), "best_epoch": int(best_epoch)}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = AMLGNN(ck["node_in"], ck["edge_in"], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck

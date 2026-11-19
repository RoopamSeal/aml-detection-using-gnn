"""Train/evaluate an edge-aware GAT for AML account classification."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

from dataset import download_kaggle_dataset, locate_transaction_csv, load_or_build_graph
from model import AMLGAT


def require_neighbor_sampling_backend():
    """Fail early with setup guidance when PyG sampling extensions are missing."""
    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise RuntimeError(
            "NeighborLoader requires either pyg-lib or torch-sparse. Install a "
            "wheel matching your PyTorch and CUDA versions using the official "
            "PyG instructions: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(data, seed=42, val_size=0.15, test_size=0.15):
    """Stratified node split. Labels are used only for split stratification."""
    ids = np.arange(data.num_nodes)
    labels = data.y.cpu().numpy().astype(int)
    train_val, test_ids = train_test_split(
        ids, test_size=test_size, random_state=seed, stratify=labels
    )
    relative_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val, test_size=relative_val, random_state=seed,
        stratify=labels[train_val]
    )
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_ids] = True
    data.val_mask[val_ids] = True
    data.test_mask[test_ids] = True
    return data


def metrics_from_logits(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y_true = labels.detach().cpu().numpy().astype(int).reshape(-1)
    pred = (probs >= threshold).astype(int)
    result = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "average_precision_pr_auc": average_precision_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    result["roc_auc"] = (
        roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else float("nan")
    )
    return result


@torch.no_grad()
def evaluate(model, data, loader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        seed_count = int(batch.batch_size)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)[:seed_count]
        all_logits.append(logits.cpu())
        all_labels.append(batch.y[:seed_count].cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return metrics_from_logits(logits, labels, threshold)


def main(args):
    require_neighbor_sampling_backend()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    source = args.csv
    if source is None:
        source = download_kaggle_dataset()
    csv_path = locate_transaction_csv(source, args.filename)
    data = load_or_build_graph(
        csv_path, args.graph_path, chunksize=args.chunksize,
        force_rebuild=args.rebuild, make_undirected=not args.directed,
    )
    data = make_splits(data, args.seed, args.val_size, args.test_size)
    print(f"Device: {device}")
    print(f"Graph: {data.num_nodes:,} accounts | {data.edge_index.size(1):,} edges "
          f"| {data.x.size(1)} node features | {data.edge_attr.size(1)} edge features")
    print(f"Positive accounts: {int(data.y.sum()):,} / {data.num_nodes:,} "
          f"({100 * float(data.y.mean()):.4f}%)")

    train_loader = NeighborLoader(
        data, num_neighbors=[args.neighbors, args.neighbors],
        batch_size=args.batch_size, input_nodes=data.train_mask,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = NeighborLoader(
        data, num_neighbors=[args.neighbors, args.neighbors],
        batch_size=args.batch_size, input_nodes=data.val_mask,
        shuffle=False, num_workers=args.num_workers,
    )
    test_loader = NeighborLoader(
        data, num_neighbors=[args.neighbors, args.neighbors],
        batch_size=args.batch_size, input_nodes=data.test_mask,
        shuffle=False, num_workers=args.num_workers,
    )

    model = AMLGAT(
        in_channels=data.x.size(1), edge_dim=data.edge_attr.size(1),
        hidden_channels=args.hidden_channels, heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    n_pos = float(data.y[data.train_mask].sum())
    n_neg = float(data.train_mask.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_ap = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            seed_count = int(batch.batch_size)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)[:seed_count]
            labels = batch.y[:seed_count].float().view(-1, 1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss) * seed_count
            seen += seed_count

        val_metrics = evaluate(model, data, val_loader, device, args.threshold)
        mean_loss = total_loss / max(seen, 1)
        row = {"epoch": epoch, "train_loss": mean_loss, **val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss {mean_loss:.5f} | "
            f"val PR-AUC {val_metrics['average_precision_pr_auc']:.5f} | "
            f"val F1 {val_metrics['f1']:.5f} | "
            f"val recall {val_metrics['recall']:.5f}"
        )
        if val_metrics["average_precision_pr_auc"] > best_val_ap:
            best_val_ap = val_metrics["average_precision_pr_auc"]
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "in_channels": data.x.size(1), "edge_dim": data.edge_attr.size(1),
                    "hidden_channels": args.hidden_channels, "heads": args.heads,
                    "dropout": args.dropout,
                },
                "threshold": args.threshold,
                "epoch": epoch,
            }, args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, data, test_loader, device, args.threshold)
    print("\nHeld-out test metrics (threshold =", args.threshold, ")")
    print(json.dumps(test_metrics, indent=2))
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_epoch": checkpoint["epoch"], "history": history,
                   "test": test_metrics}, f, indent=2)
    print(f"Best checkpoint: {args.checkpoint}")
    print(f"Metrics: {args.metrics_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="CSV path or KaggleHub folder; omit to download.")
    p.add_argument("--filename", type=str, default="HI-Small_Trans.csv")
    p.add_argument("--graph-path", type=str, default="data/processed/aml_graph.pt")
    p.add_argument("--checkpoint", type=str, default="artifacts/best_gat.pt")
    p.add_argument("--metrics-path", type=str, default="artifacts/metrics.json")
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--directed", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    main(p.parse_args())

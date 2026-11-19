"""Train and evaluate the small AML GAT baseline."""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from dataset import build_graph
from model import AMLGAT

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_masks(data, seed=42):
    ids = np.arange(data.num_nodes)
    y = data.y.numpy().astype(int)
    train_ids, temp_ids = train_test_split(
        ids, test_size=0.30, random_state=seed, stratify=y
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed, stratify=y[temp_ids]
    )
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_ids] = True
    data.val_mask[val_ids] = True
    data.test_mask[test_ids] = True
    return data

def calculate_metrics(logits, labels, threshold=0.5):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }

@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    return calculate_metrics(logits[mask.to(device)], data.y[mask].to(device))

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    data = build_graph(args.csv, max_transactions=args.max_transactions, seed=args.seed)
    data = make_masks(data, args.seed)

    for name, mask in [("Train", data.train_mask), ("Validation", data.val_mask), ("Test", data.test_mask)]:
        print(f"{name:10s}: {int(mask.sum()):,} accounts | positive rate {data.y[mask].mean().item():.3%}")

    data = data.to(device)
    model = AMLGAT(
        in_channels=data.x.size(1),
        edge_dim=data.edge_attr.size(1),
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    positives = data.y[data.train_mask].sum()
    negatives = data.train_mask.sum() - positives
    pos_weight = (negatives / positives.clamp(min=1)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_ap = -1.0
    best_epoch = 0
    patience_left = args.patience
    history = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        val = evaluate(model, data, data.val_mask, device)
        row = {"epoch": epoch, "train_loss": float(loss.item()), **val}
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={loss.item():.4f} "
            f"| val PR-AUC={val['pr_auc']:.4f} | val F1={val['f1']:.4f}"
        )

        if val["pr_auc"] > best_val_ap:
            best_val_ap = val["pr_auc"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_channels": data.x.size(1),
                "edge_dim": data.edge_attr.size(1),
                "hidden_channels": args.hidden_channels,
                "dropout": args.dropout,
                "epoch": epoch,
            }, Path(args.output_dir) / "best_gat.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    checkpoint = torch.load(Path(args.output_dir) / "best_gat.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, data, data.test_mask, device)

    results = {
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_val_ap,
        "test_metrics": test_metrics,
        "history": history,
        "note": "Account labels are derived from Is Laundering on sampled transactions; this is a transductive educational baseline."
    }
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Training loss")
    plt.tight_layout(); plt.savefig(Path(args.output_dir) / "training_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["pr_auc"] for r in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Validation PR-AUC"); plt.title("Validation PR-AUC")
    plt.tight_layout(); plt.savefig(Path(args.output_dir) / "validation_pr_auc.png", dpi=150); plt.close()

    print("\nHeld-out test results:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved checkpoint and metrics in {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Optional path to HI-Small_Trans.csv")
    parser.add_argument("--max-transactions", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())

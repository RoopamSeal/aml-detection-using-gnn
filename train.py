"""Mini-batch, chronological AML transaction classification using an edge-aware GAT."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)
from dataset import load_aml_graph, NeighborhoodSampler
from model import TransactionGAT


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_batch(g, sampler, ids, device, excluded=None):
    seeds = np.concatenate([g.src[ids], g.dst[ids]])
    nodes, ei, ea, mp = sampler.sample(seeds, excluded_txids=excluded)
    qs = np.asarray([mp[int(v)] for v in g.src[ids]], dtype=np.int64)
    qd = np.asarray([mp[int(v)] for v in g.dst[ids]], dtype=np.int64)
    return (
        torch.as_tensor(g.x[nodes], dtype=torch.float32, device=device),
        torch.as_tensor(ei, dtype=torch.long, device=device),
        torch.as_tensor(ea, dtype=torch.float32, device=device),
        torch.as_tensor(qs, dtype=torch.long, device=device),
        torch.as_tensor(qd, dtype=torch.long, device=device),
        torch.as_tensor(g.edge_attr[ids], dtype=torch.float32, device=device),
        torch.as_tensor(g.y[ids], dtype=torch.float32, device=device),
    )


def iter_batches(indices, size, shuffle=False):
    ids = np.asarray(indices).copy()
    if shuffle: np.random.shuffle(ids)
    for i in range(0, len(ids), size): yield ids[i:i+size]


def evaluate(model, g, sampler, indices, batch_size, device, criterion=None):
    model.eval(); probs, labels = [], []; loss_sum = 0.; count = 0
    with torch.no_grad():
        for ids in iter_batches(indices, batch_size):
            x, ei, ea, qs, qd, qa, y = make_batch(g, sampler, ids, device)
            logits = model(x, ei, ea, qs, qd, qa)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * len(ids); count += len(ids)
            probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y.cpu().numpy())
    yt, yp = np.concatenate(labels).astype(int), np.concatenate(probs)
    yh = (yp >= .5).astype(int)
    out = {
        "precision": float(precision_score(yt, yh, zero_division=0)),
        "recall": float(recall_score(yt, yh, zero_division=0)),
        "f1": float(f1_score(yt, yh, zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yh).tolist(),
        "roc_auc": float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else None,
        "pr_auc": float(average_precision_score(yt, yp)) if len(np.unique(yt)) == 2 else None,
    }
    if criterion is not None and count: out["loss"] = loss_sum / count
    return out


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)
    g = load_aml_graph(args.csv, args.max_transactions, args.train_ratio, args.val_ratio)
    sampler = NeighborhoodSampler(g.x.shape[0], g.context_src, g.context_dst,
        g.context_attr, g.context_txid, fanout=args.neighbors, num_hops=2, seed=args.seed)

    pos = float(g.y[g.train_idx].sum()); neg = len(g.train_idx) - pos
    if pos == 0: raise ValueError("No positive laundering examples in training split.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    model = TransactionGAT(g.x.shape[1], g.edge_attr.shape[1],
                           args.hidden_channels, args.dropout).to(device)
    # SGD is intentionally used as requested.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    best_score = -1.; best_path = outdir / "best_gat.pt"; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.; seen = 0
        for ids in iter_batches(g.train_idx, args.train_batch_size, shuffle=True):
            # Exclude target training edges from their own sampled graph context.
            x, ei, ea, qs, qd, qa, y = make_batch(g, sampler, ids, device, excluded=ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, ei, ea, qs, qd, qa)
            loss = criterion(logits, y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step(); total_loss += loss.item() * len(ids); seen += len(ids)
        train_loss = total_loss / max(seen, 1)
        val = evaluate(model, g, sampler, g.val_idx, args.test_batch_size, device, criterion)
        score = val["pr_auc"] if val["pr_auc"] is not None else -1.
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val.get("loss"), "val_pr_auc": val["pr_auc"],
                        "val_roc_auc": val["roc_auc"], "val_f1": val["f1"]})
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.5f} | "
              f"val_loss={val['loss']:.5f} | val_PR-AUC={score:.5f} | val_ROC-AUC={val['roc_auc']}")
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_metrics": val, "config": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, g, sampler, g.test_idx, args.test_batch_size, device)
    result = {"best_epoch": int(checkpoint["epoch"]),
              "best_validation_metrics": checkpoint["validation_metrics"],
              "test_metrics": test, "config": vars(args),
              "protocol": "Chronological transaction split; training-period graph only; each training batch's target edges are excluded from its sampled context."}
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    print("\nTEST METRICS")
    for k, v in test.items(): print(f"{k}: {v}")
    print(f"\nSaved checkpoint: {best_path}\nSaved metrics: {outdir/'metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/kaggle/HI-Small_Trans.csv")
    p.add_argument("--max-transactions", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--optimizer", choices=["SGD"], default="SGD")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=.70)
    p.add_argument("--val-ratio", type=float, default=.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())

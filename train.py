"""Train the edge-aware GNN on IBM AML HI-Small (Trans).

Weighted BCE (pos_weight = sqrt(neg/pos)), AdamW + cosine schedule, ~20 mini-batch steps per epoch
(time-contiguous chunks of ~150k transactions, boundaries randomly shifted every epoch),
best epoch chosen by validation PR-AUC, early stopping on validation PR-AUC.

    python train.py                                   # full HI-Small, notebook defaults
    python train.py --max-edges 500000 --epochs 3     # quick smoke test
"""
import argparse
import json
import os
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")  # headless (Codespaces has no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from config import CHECKPOINT_NAME, OUTPUT_DIR, Config
from dataset import AMLData, prepare_data
from model import AMLGNN, count_parameters, predict, save_checkpoint


def parse_args():
    d = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=None, help="CSV path. Default: data/HI-Small_Trans.csv (downloaded if missing)")
    ap.add_argument("--max-edges", type=int, default=d.max_edges, help="Use only the first N transactions (by time). Default: all")
    ap.add_argument("--chunk", type=int, default=d.chunk, help="Transactions per training sub-graph")
    ap.add_argument("--hidden", type=int, default=d.hidden)
    ap.add_argument("--layers", type=int, default=d.layers)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using data/cache/")
    return ap.parse_args()


def train(model: nn.Module, data: AMLData, cfg: Config, device, ckpt_path: str):
    n_tr, n_va = data.n_tr, data.n_va
    pos = data.Y[:n_tr].sum().item()
    if pos == 0:
        raise ValueError("No laundering transactions in the training period. Use more data (larger --max-edges).")
    pos_w = torch.tensor(((n_tr - pos) / pos) ** 0.5, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    y_va = data.Y[n_tr:n_va].numpy()
    hist_log, best_ap, best_state, best_ep, bad = [], -1, None, 0, 0

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        off = np.random.randint(0, cfg.chunk)  # random chunk boundaries every epoch
        bounds = sorted(set([0] + list(range(off, n_tr, cfg.chunk)) + [n_tr]))
        spans = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1000]
        np.random.shuffle(spans)
        losses = []
        for a, b in spans:
            x, ei, ea, y = data.make_chunk(a, b, device)
            loss = F.binary_cross_entropy_with_logits(model(x, ei, ea), y, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        ap = average_precision_score(y_va, predict(model, data, n_tr, n_va, cfg.chunk, device))
        hist_log.append((float(np.mean(losses)), float(ap)))
        if ap > best_ap:
            best_ap, best_ep, bad = ap, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # save immediately, so an interrupted run (e.g. Codespace timeout) still leaves a usable model
            save_checkpoint(ckpt_path, best_state, asdict(cfg), data.NODE_IN, data.edge_dim, best_ap, best_ep)
        else:
            bad += 1
        print(f"epoch {ep:02d} | steps {len(spans)} | loss {np.mean(losses):.4f} | "
              f"val PR-AUC {ap:.4f} | {time.time() - t0:.0f}s", flush=True)
        if bad >= cfg.patience:
            print("early stop")
            break

    model.load_state_dict(best_state)
    return hist_log, best_ap, best_ep


def plot_history(hist_log, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([h[0] for h in hist_log])
    ax[0].set_title("Train loss")
    ax[1].plot([h[1] for h in hist_log])
    ax[1].set_title("Validation PR-AUC")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config(max_edges=args.max_edges, chunk=args.chunk, hidden=args.hidden, layers=args.layers,
                 dropout=args.dropout, epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("config:", asdict(cfg))

    data = prepare_data(args.data_path, cfg.max_edges, with_tree=False, use_cache=not args.no_cache)

    model = AMLGNN(data.NODE_IN, data.edge_dim, cfg.hidden, cfg.layers, cfg.dropout).to(device)
    print("parameters:", count_parameters(model))

    ckpt_path = os.path.join(args.output_dir, CHECKPOINT_NAME)
    t0 = time.time()
    hist_log, best_ap, best_ep = train(model, data, cfg, device, ckpt_path)
    print(f"\nbest epoch {best_ep} | best val PR-AUC {best_ap:.4f} | total {(time.time() - t0) / 60:.1f} min")

    plot_history(hist_log, os.path.join(args.output_dir, "training_curves.png"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({"config": asdict(cfg), "best_epoch": best_ep, "best_val_pr_auc": best_ap,
                   "history": [{"epoch": i + 1, "train_loss": l, "val_pr_auc": a} for i, (l, a) in enumerate(hist_log)]},
                  f, indent=2)
    print(f"Saved: {ckpt_path}, training_curves.png, history.json in {args.output_dir}/")
    print(f"Next: python evaluate.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()

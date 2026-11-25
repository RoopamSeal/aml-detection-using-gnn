"""Evaluate a trained checkpoint on the held-out (last 20%) test period.

The decision threshold is chosen on validation (max F1) and applied unchanged to test.
Also runs the retroactive-flagging post-processing and, optionally, the tree baseline.

    python evaluate.py                       # GNN + retroactive flagging
    python evaluate.py --tree-baseline       # additionally the tree baseline (~2 min)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless (Codespaces has no display)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from config import CHECKPOINT_NAME, OUTPUT_DIR
from dataset import prepare_data
from metrics import REFERENCE_NOTE, WINDOWS, best_threshold, prf, report, retro, tune_retro
from model import load_checkpoint, predict


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(OUTPUT_DIR, CHECKPOINT_NAME))
    ap.add_argument("--data-path", default=None, help="CSV path. Default: data/HI-Small_Trans.csv")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--tree-baseline", action="store_true", help="Also train/evaluate the tree baseline on the same features")
    ap.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using data/cache/")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, ck = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (best epoch {ck['best_epoch']}, val PR-AUC {ck['best_val_pr_auc']:.4f})")

    # Rebuild the exact same data/split as training (same max_edges)
    data = prepare_data(args.data_path, cfg["max_edges"], with_tree=args.tree_baseline, use_cache=not args.no_cache)
    if data.edge_dim != ck["edge_in"]:
        raise ValueError(f"Edge feature dim mismatch: data has {data.edge_dim}, checkpoint expects {ck['edge_in']}. "
                         "Was the checkpoint trained on a different dataset/max_edges?")
    n_tr, n_va, n = data.n_tr, data.n_va, data.n
    results = {}

    # ---- (Optional) tree baseline ----
    if args.tree_baseline:
        from baseline import run_tree_baseline
        results["Trees (same features)"] = run_tree_baseline(data)
        print(pd.Series(results["Trees (same features)"]).round(4).to_string(), "\n")
        data.tree_X = None  # free memory

    # ---- GNN on the test period ----
    pv = predict(model, data, n_tr, n_va, cfg["chunk"], device)
    pt = predict(model, data, n_va, n, cfg["chunk"], device)
    y_va, y_te = data.Y[n_tr:n_va].numpy(), data.Y[n_va:].numpy()

    thr = best_threshold(y_va, pv)
    name = "GNN (edge-aware GINE)"
    results[name] = report(y_te, pt, thr)
    pred = (pt >= thr).astype(int)

    print(f"Threshold (from validation): {thr:.4f}\n")
    print(pd.Series(results[name]).round(4).to_string(), "\n")
    cls_report = classification_report(y_te, pred, target_names=["Legit", "Laundering"], digits=4, zero_division=0)
    print(cls_report)

    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    plt.xticks([0, 1], ["Legit", "Launder"])
    plt.yticks([0, 1], ["Legit", "Launder"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    pr, rc, _ = precision_recall_curve(y_te, pt)
    plt.figure(figsize=(5, 4))
    plt.plot(rc, pr)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pr_curve.png"), dpi=150)
    plt.close()

    # ---- Retroactive flagging (delayed alerts), tuned on validation ----
    best = tune_retro(data, y_va, pv, thr)
    print(f"chosen on validation: look-back {best[1]}, earlier score >= {best[2]} (val F1 {best[0]:.4f})")
    flag_te = retro(data.SND, data.RCV, data.SECS, n_va, n, pt, thr, WINDOWS[best[1]], best[2])
    results["GNN + retroactive flagging (delayed)"] = prf(y_te, flag_te)

    # ---- Summary ----
    summary = pd.DataFrame(results).T.round(4)
    print("\n=== Summary (test set) ===")
    print(summary.to_string())
    print(REFERENCE_NOTE)

    summary.to_csv(os.path.join(args.output_dir, "summary.csv"))
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Threshold (from validation): {thr:.4f}\n\n{cls_report}")
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"threshold": float(thr), "retro_window": best[1], "retro_min_score": best[2],
                   "results": {k: {m: (None if pd.isna(v) else float(v)) for m, v in r.items()} for k, r in results.items()}},
                  f, indent=2)
    print(f"\nSaved summary.csv, metrics.json, classification_report.txt, confusion_matrix.png, pr_curve.png in {args.output_dir}/")


if __name__ == "__main__":
    main()

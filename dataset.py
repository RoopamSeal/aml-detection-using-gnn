"""Dataset module: download -> load -> causal features -> edge tensors -> time-contiguous chunks.

Dataset: IBM "Transactions for Anti Money Laundering (AML)", HI-Small variant (HI-Small_Trans.csv)
         https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Accounts are nodes, transactions are directed edges, the label is `Is Laundering`.

Quick usage from the command line:
    python dataset.py                 # download (if needed) and print basic dataset statistics
    python dataset.py --prepare       # also build all features/tensors and cache them under data/cache/
"""
import argparse
import gc
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import torch

from config import (CACHE_DIR, COLS, DATA_DIR, DATASET_FILE, KAGGLE_DATASET, KAGGLE_INPUT_PATH, KAGGLE_URL)
from features import BASIC, NEW_FEATURES, add_features

CACHE_VERSION = 1  # bump if the feature/tensor code changes, so stale caches are ignored


# --------------------------------------------------------------------------- #
# 1. Get the data
# --------------------------------------------------------------------------- #
def download_dataset(data_dir: str = DATA_DIR) -> str:
    """Download HI-Small_Trans.csv from Kaggle with the Kaggle CLI. Needs Kaggle API credentials."""
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, DATASET_FILE)

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("The `kaggle` CLI was not found. Run: pip install -r requirements.txt")

    cmd = [kaggle, "datasets", "download", "-d", KAGGLE_DATASET, "-f", DATASET_FILE, "-p", data_dir, "--unzip"]
    print("Downloading:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Kaggle download failed. Make sure Kaggle credentials are configured, either:\n"
            "  * Codespaces secrets KAGGLE_USERNAME and KAGGLE_KEY (then restart the Codespace), or\n"
            "  * a ~/.kaggle/kaggle.json file (chmod 600).\n"
            f"Alternatively download {DATASET_FILE} manually from {KAGGLE_URL} and place it at {target}."
        ) from e

    zip_path = target + ".zip"  # older kaggle versions may leave a zip behind
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    if not os.path.exists(target):
        raise FileNotFoundError(f"Expected {target} after download but it does not exist.")
    return target


def resolve_data_path(data_path: Optional[str] = None) -> str:
    """Explicit path > Kaggle input path > data/HI-Small_Trans.csv > download from Kaggle."""
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"--data-path not found: {data_path}")
        return data_path
    if os.path.exists(KAGGLE_INPUT_PATH):
        return KAGGLE_INPUT_PATH
    local = os.path.join(DATA_DIR, DATASET_FILE)
    if os.path.exists(local):
        return local
    return download_dataset(DATA_DIR)


def load_transactions(path: str, max_edges: Optional[int] = None) -> pd.DataFrame:
    """Read the raw CSV, sort chronologically (stable), optionally keep only the first `max_edges` rows."""
    tx = pd.read_csv(path, header=0, names=COLS, parse_dates=["ts"],
                     dtype={"from_bank": str, "to_bank": str, "from_acct": str, "to_acct": str})
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    if max_edges:
        tx = tx.iloc[:max_edges].reset_index(drop=True)
    return tx


# --------------------------------------------------------------------------- #
# 2. Prepared data container (+ time-contiguous sub-graphs)
# --------------------------------------------------------------------------- #
@dataclass
class AMLData:
    EA: torch.Tensor      # [n, edge_dim] standardised edge features (train-only statistics)
    Y: torch.Tensor       # [n] laundering label
    AMT: torch.Tensor     # [n, 2] raw (amt_paid, amt_recv), used for node features
    EI: torch.Tensor      # [2, n] (sender, receiver) account indices
    SND: np.ndarray       # sender codes   (kept for retroactive flagging)
    RCV: np.ndarray       # receiver codes (kept for retroactive flagging)
    SECS: np.ndarray      # seconds since first transaction
    n: int
    n_tr: int             # train = [0, n_tr)
    n_va: int             # val   = [n_tr, n_va), test = [n_va, n)
    tree_X: Optional[pd.DataFrame] = None   # only filled when the tree baseline is requested

    NODE_IN: ClassVar[int] = 4

    @property
    def edge_dim(self) -> int:
        return self.EA.shape[1]

    def make_chunk(self, lo: int, hi: int, device):
        """Treat transactions [lo, hi) as their own graph (accounts re-indexed locally).

        Node features are label-free structural features: log in/out degree and log amount
        sent/received inside the chunk, standardised within the chunk.
        """
        ei = self.EI[:, lo:hi]
        nodes, inv = torch.unique(ei, return_inverse=True)
        m = len(nodes)
        out_deg = torch.bincount(inv[0], minlength=m).float()
        in_deg = torch.bincount(inv[1], minlength=m).float()
        out_amt = torch.zeros(m).index_add_(0, inv[0], self.AMT[lo:hi, 0])
        in_amt = torch.zeros(m).index_add_(0, inv[1], self.AMT[lo:hi, 1])
        x = torch.stack([out_deg, in_deg, out_amt, in_amt], 1).log1p()
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        return x.to(device), inv.to(device), self.EA[lo:hi].to(device), self.Y[lo:hi].to(device)


# --------------------------------------------------------------------------- #
# 3. Build tabular features and edge tensors
# --------------------------------------------------------------------------- #
def _build(path: str, max_edges: Optional[int], with_tree: bool) -> AMLData:
    t0 = time.time()
    tx = load_transactions(path, max_edges)
    print(f"{len(tx):,} transactions, {tx.label.sum():,} laundering ({tx.label.mean():.3%})")

    df = add_features(tx)
    del tx
    df["hour"], df["dow"] = df.ts.dt.hour, df.ts.dt.day_of_week
    df["log_amt_paid"] = np.log1p(df.amt_paid)
    df["cross_cur"] = (df.cur_recv != df.cur_paid).astype(np.int8)
    df["cross_bank"] = (df.from_bank != df.to_bank).astype(np.int8)
    df["self_xfer"] = (df.from_acct == df.to_acct).astype(np.int8)
    n = len(df)
    n_tr, n_va = int(.6 * n), int(.8 * n)  # chronological 60/20/20 split
    print(f"features built in {time.time() - t0:.0f}s | split sizes: {n_tr:,} / {n_va - n_tr:,} / {n - n_va:,}")
    print("laundering in train/val/test:",
          int(df.label[:n_tr].sum()), int(df.label[n_tr:n_va].sum()), int(df.label[n_va:].sum()))

    # Feature table for the (optional) tree baseline
    tree_X = None
    if with_tree:
        tree_X = df[BASIC + NEW_FEATURES].copy()
        for c in ["cur_recv", "cur_paid", "fmt"]:
            tree_X[c] = tree_X[c].astype("category")

    # ---- Edge features: log1p on counts/ages/gaps, standardise with TRAIN-ONLY stats, one-hot categoricals ----
    basic_num = pd.DataFrame({"hour": df.hour / 23.0, "dow": df.dow / 6.0, "log_amt_paid": df.log_amt_paid,
                              "cross_cur": df.cross_cur.astype(float), "cross_bank": df.cross_bank.astype(float),
                              "self_xfer": df.self_xfer.astype(float)})
    hist = df[NEW_FEATURES].astype(np.float64)
    for c in ["hrs_since_last_pair", "hrs_since_last_sender_tx"]:
        hist[c + "_isna"] = hist[c].isna().astype(float)
        hist[c] = hist[c].fillna(0)
    hist = np.log1p(hist.clip(lower=0))
    num = pd.concat([basic_num, hist], axis=1)
    mu, sd = num.iloc[:n_tr].mean(), num.iloc[:n_tr].std() + 1e-6
    num = ((num - mu) / sd).clip(-5, 5)
    cat = pd.get_dummies(df[["fmt", "cur_recv", "cur_paid"]].astype(str), dtype=float)

    EA = torch.tensor(pd.concat([num, cat], axis=1).to_numpy(np.float32))
    Y = torch.tensor(df.label.to_numpy(), dtype=torch.float)
    AMT = torch.tensor(df[["amt_paid", "amt_recv"]].to_numpy(np.float32))

    codes, _ = pd.factorize(pd.concat([df.from_acct, df.to_acct], ignore_index=True))
    SND, RCV = codes[:n].astype(np.int64), codes[n:].astype(np.int64)  # kept for retroactive flagging
    EI = torch.tensor(np.vstack([SND, RCV]))
    SECS = ((df.ts - df.ts.min()).dt.total_seconds()).to_numpy(np.int64)
    print(f"{codes.max() + 1:,} accounts | edge feature dim = {EA.shape[1]}")
    del basic_num, hist, num, cat, codes, df
    gc.collect()

    return AMLData(EA=EA, Y=Y, AMT=AMT, EI=EI, SND=SND, RCV=RCV, SECS=SECS,
                   n=n, n_tr=n_tr, n_va=n_va, tree_X=tree_X)


def prepare_data(data_path: Optional[str] = None, max_edges: Optional[int] = None,
                 with_tree: bool = False, use_cache: bool = True) -> AMLData:
    """Return the fully prepared dataset. Features take ~1-2 min for all 5M rows, so they are cached on disk.

    The cache is keyed on `max_edges` only. Delete data/cache/ (or pass use_cache=False) if you change the CSV.
    """
    tag = f"edges{max_edges}" if max_edges else "edgesall"
    cache_pt = os.path.join(CACHE_DIR, f"{tag}_v{CACHE_VERSION}.pt")
    cache_tree = os.path.join(CACHE_DIR, f"{tag}_v{CACHE_VERSION}_tree.pkl")

    if use_cache and os.path.exists(cache_pt) and (not with_tree or os.path.exists(cache_tree)):
        print(f"Loading cached features: {cache_pt}")
        d = torch.load(cache_pt, map_location="cpu")
        data = AMLData(EA=d["EA"], Y=d["Y"], AMT=d["AMT"], EI=d["EI"],
                       SND=d["SND"].numpy(), RCV=d["RCV"].numpy(), SECS=d["SECS"].numpy(),
                       n=int(d["n"]), n_tr=int(d["n_tr"]), n_va=int(d["n_va"]))
        if with_tree:
            data.tree_X = pd.read_pickle(cache_tree)
        print(f"{data.n:,} transactions | edge feature dim = {data.edge_dim} | "
              f"split sizes: {data.n_tr:,} / {data.n_va - data.n_tr:,} / {data.n - data.n_va:,}")
        return data

    path = resolve_data_path(data_path)
    print("Using:", path)
    data = _build(path, max_edges, with_tree)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"EA": data.EA, "Y": data.Y, "AMT": data.AMT, "EI": data.EI,
                    "SND": torch.from_numpy(data.SND), "RCV": torch.from_numpy(data.RCV),
                    "SECS": torch.from_numpy(data.SECS),
                    "n": data.n, "n_tr": data.n_tr, "n_va": data.n_va}, cache_pt)
        if with_tree:
            data.tree_X.to_pickle(cache_tree)
        print(f"Cached features -> {cache_pt}")
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download the HI-Small dataset and (optionally) pre-build features.")
    ap.add_argument("--data-path", default=None, help="Use this CSV instead of downloading")
    ap.add_argument("--max-edges", type=int, default=None, help="Keep only the first N transactions (by time)")
    ap.add_argument("--prepare", action="store_true", help="Also build features/tensors and cache them")
    ap.add_argument("--with-tree", action="store_true", help="With --prepare: also cache the tree-baseline features")
    args = ap.parse_args()

    if args.prepare:
        d = prepare_data(args.data_path, args.max_edges, with_tree=args.with_tree)
    else:
        p = resolve_data_path(args.data_path)
        tx = load_transactions(p, args.max_edges)
        print(f"{p}: {len(tx):,} transactions, {int(tx.label.sum()):,} laundering ({tx.label.mean():.3%})")
        print(f"time range: {tx.ts.min()} -> {tx.ts.max()}")

"""Chronological transaction graph and lightweight neighborhood sampler."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class AMLGraph:
    x: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    context_src: np.ndarray
    context_dst: np.ndarray
    context_attr: np.ndarray
    context_txid: np.ndarray


def load_aml_graph(csv_path, max_transactions=100_000, train_ratio=.70, val_ratio=.15):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    cols = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Is Laundering"]
    df = pd.read_csv(path, usecols=cols, nrows=max_transactions)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    df["Is Laundering"] = pd.to_numeric(df["Is Laundering"], errors="coerce")
    df = df.dropna(subset=cols).copy()
    df = df[df["Is Laundering"].isin([0, 1])]
    df["Is Laundering"] = df["Is Laundering"].astype(int)
    df["Amount Paid"] = df["Amount Paid"].clip(lower=0)
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    if n < 100 or a == 0 or b <= a or b >= n:
        raise ValueError(f"Not enough usable transactions for split (n={n}).")

    # One graph node per bank/account pair.
    sk = df["From Bank"].astype(str) + "::" + df["Account"].astype(str)
    rk = df["To Bank"].astype(str) + "::" + df["Account.1"].astype(str)
    codes, _ = pd.factorize(pd.concat([sk, rk], ignore_index=True), sort=True)
    src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    nn = int(codes.max()) + 1
    sb = pd.to_numeric(df["From Bank"], errors="coerce").fillna(0).to_numpy()
    rb = pd.to_numeric(df["To Bank"], errors="coerce").fillna(0).to_numpy()
    bank = np.zeros(nn, dtype=np.float64)
    np.maximum.at(bank, src, sb); np.maximum.at(bank, dst, rb)

    y = df["Is Laundering"].to_numpy(np.float32)
    amount = df["Amount Paid"].to_numpy(np.float64)
    ts = df["Timestamp"].astype("int64").to_numpy(np.float64) / 1e9
    train_idx = np.arange(a, dtype=np.int64)
    val_idx = np.arange(a, b, dtype=np.int64)
    test_idx = np.arange(b, n, dtype=np.int64)

    # Static bank identifier feature only. We deliberately avoid account activity
    # aggregates here because they could contain the current target transaction.
    bankf = np.log1p(np.maximum(bank, 0))
    bankf /= max(float(bankf.max()), 1.)
    x = bankf.reshape(-1, 1).astype(np.float32)

    log_amt = np.log1p(amount)
    amt_scale = max(float(log_amt[train_idx].max()), 1.)
    t0 = float(ts[train_idx].min())
    tscale = max(float(ts[train_idx].max() - t0), 1.)
    edge_attr = np.column_stack([log_amt / amt_scale, (ts - t0) / tscale]).astype(np.float32)

    # Historical message-passing graph: training edges in both directions.
    csrc = np.concatenate([src[train_idx], dst[train_idx]])
    cdst = np.concatenate([dst[train_idx], src[train_idx]])
    cattr = np.concatenate([edge_attr[train_idx], edge_attr[train_idx]], axis=0)
    ctxid = np.concatenate([train_idx, train_idx])

    print(f"Transactions: {n:,} | account nodes: {nn:,}")
    for name, ids in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        pos = int(y[ids].sum())
        print(f"{name:10s}: {len(ids):,} rows; laundering={pos:,} ({pos/len(ids):.4%})")
        if pos == 0 or pos == len(ids):
            raise ValueError(f"{name} split has only one class. Increase --max-transactions or inspect date coverage.")

    return AMLGraph(x, src, dst, edge_attr, y, train_idx, val_idx, test_idx,
                    csrc, cdst, cattr, ctxid)


class NeighborhoodSampler:
    """Two-hop fanout sampler; does not require pyg-lib or torch-sparse."""
    def __init__(self, num_nodes, edge_src, edge_dst, edge_attr, edge_txid,
                 fanout=5, num_hops=2, seed=42):
        self.edge_src, self.edge_dst = edge_src, edge_dst
        self.edge_attr, self.edge_txid = edge_attr, edge_txid
        self.fanout, self.num_hops = fanout, num_hops
        self.rng = np.random.default_rng(seed)
        self.adj = [[] for _ in range(num_nodes)]
        for eid, (u, v) in enumerate(zip(edge_src, edge_dst)):
            self.adj[int(u)].append((int(v), eid))

    def sample(self, seeds, excluded_txids=None):
        excluded = set() if excluded_txids is None else set(map(int, excluded_txids))
        nodes, frontier, eids = set(map(int, seeds)), set(map(int, seeds)), set()
        for _ in range(self.num_hops):
            nxt = set()
            for u in frontier:
                cand = [p for p in self.adj[u] if int(self.edge_txid[p[1]]) not in excluded]
                if len(cand) > self.fanout:
                    take = self.rng.choice(len(cand), self.fanout, replace=False)
                    cand = [cand[int(i)] for i in take]
                for v, eid in cand:
                    eids.add(eid)
                    if v not in nodes: nxt.add(v)
            nodes.update(nxt); frontier = nxt
            if not frontier: break
        local_nodes = np.asarray(sorted(nodes), dtype=np.int64)
        mapping = {int(v): i for i, v in enumerate(local_nodes)}
        if eids:
            ids = np.asarray(sorted(eids), dtype=np.int64)
            ei = np.vstack([
                [mapping[int(v)] for v in self.edge_src[ids]],
                [mapping[int(v)] for v in self.edge_dst[ids]],
            ]).astype(np.int64)
            ea = self.edge_attr[ids]
        else:
            ei = np.empty((2, 0), dtype=np.int64)
            ea = np.empty((0, self.edge_attr.shape[1]), dtype=np.float32)
        return local_nodes, ei, ea, mapping

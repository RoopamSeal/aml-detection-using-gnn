"""Load a small IBM AML transaction sample and convert it into a PyG graph."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import kagglehub
from torch_geometric.data import Data

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
FILENAME = "HI-Small_Trans.csv"

def download_dataset():
    """Download the Kaggle dataset. Configure Kaggle credentials in Codespaces first."""
    folder = Path(kagglehub.dataset_download(DATASET))
    matches = list(folder.rglob(FILENAME))
    if not matches:
        matches = list(folder.rglob("*Small*Trans.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find {FILENAME} under {folder}")
    return matches[0]

def build_graph(csv_path=None, max_transactions=25000, seed=42):
    """
    Build an account graph from the first max_transactions rows of HI-Small_Trans.csv.

    Nodes = bank/account pairs.
    Edges = transactions (sender -> receiver).
    Account label = 1 if it is an endpoint of a transaction labelled Is Laundering=1.

    The small row limit is intentional for a quick Codespaces experiment.
    """
    csv_path = Path(csv_path) if csv_path else download_dataset()
    usecols = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Amount Paid", "Is Laundering"
    ]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_transactions)
    df = df.dropna(subset=usecols).copy()
    if df.empty:
        raise ValueError("No usable rows found in the selected CSV sample.")

    # Keep account identifiers unique across banks.
    src_key = df["From Bank"].astype(str) + "::" + df["Account"].astype(str)
    dst_key = df["To Bank"].astype(str) + "::" + df["Account.1"].astype(str)
    all_keys = pd.Index(pd.concat([src_key, dst_key], ignore_index=True).unique())
    node_map = {key: idx for idx, key in enumerate(all_keys)}
    src = src_key.map(node_map).to_numpy(dtype=np.int64)
    dst = dst_key.map(node_map).to_numpy(dtype=np.int64)
    n_nodes = len(all_keys)

    paid = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    received = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    labels = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).to_numpy().astype(int)

    # Node features: normalized bank ID, log mean paid/received, log transaction counts.
    bank_by_key = {}
    for bank, account in zip(df["From Bank"].astype(str), df["Account"].astype(str)):
        bank_by_key.setdefault(bank + "::" + account, float(bank) if bank.replace(".", "", 1).isdigit() else 0.0)
    for bank, account in zip(df["To Bank"].astype(str), df["Account.1"].astype(str)):
        bank_by_key.setdefault(bank + "::" + account, float(bank) if bank.replace(".", "", 1).isdigit() else 0.0)
    bank_values = np.array([bank_by_key.get(k, 0.0) for k in all_keys], dtype=np.float32)
    bank_values = (bank_values - bank_values.min()) / max(float(bank_values.max() - bank_values.min()), 1.0)

    paid_sum = np.bincount(src, weights=paid, minlength=n_nodes)
    recv_sum = np.bincount(dst, weights=received, minlength=n_nodes)
    sent_count = np.bincount(src, minlength=n_nodes)
    recv_count = np.bincount(dst, minlength=n_nodes)
    paid_mean = paid_sum / np.maximum(sent_count, 1)
    recv_mean = recv_sum / np.maximum(recv_count, 1)

    x = np.column_stack([
        bank_values,
        np.log1p(paid_mean),
        np.log1p(recv_mean),
        np.log1p(sent_count),
        np.log1p(recv_count),
    ]).astype(np.float32)

    # Edge attributes: normalized time and log transaction amounts.
    timestamps = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    seconds = timestamps.astype("int64").to_numpy(dtype=np.float64) / 1e9
    seconds[~timestamps.notna().to_numpy()] = np.nan
    lo, hi = np.nanmin(seconds), np.nanmax(seconds)
    time_feature = np.nan_to_num((seconds - lo) / max(hi - lo, 1.0), nan=0.0)
    edge_attr = np.column_stack([
        time_feature,
        np.log1p(paid),
        np.log1p(received),
    ]).astype(np.float32)

    # Account-level proxy labels: either endpoint of a flagged transaction.
    y = np.zeros(n_nodes, dtype=np.float32)
    positive_edges = labels == 1
    y[src[positive_edges]] = 1.0
    y[dst[positive_edges]] = 1.0

    # Bidirectional message passing; each transaction's features are copied to its reverse edge.
    edge_index = np.vstack([src, dst])
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        num_nodes=n_nodes,
    )
    data.num_transactions = int(len(df))
    data.num_laundering_transactions = int(labels.sum())
    data.account_keys = list(all_keys)
    print(f"Loaded {len(df):,} transactions")
    print(f"Graph: {n_nodes:,} accounts | {data.edge_index.size(1):,} directed message edges")
    print(f"Positive accounts: {int(y.sum()):,} ({100*y.mean():.3f}%)")
    return data

if __name__ == "__main__":
    graph = build_graph()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    torch.save(graph, "data/processed/aml_graph.pt")
    print("Saved data/processed/aml_graph.pt")

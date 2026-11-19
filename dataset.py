"""
Build a transaction graph from the IBM AML Kaggle dataset.

Accounts are nodes; transactions are directed edges (sender -> receiver).
The target is node-level: an account is positive if it appears as either endpoint
of at least one transaction whose Is Laundering flag is 1. This matches the
label construction in the supplied reference notebook.

For large CSVs we scan in chunks (twice) instead of loading the entire CSV into
a pandas DataFrame. The final graph still has to fit in RAM for this PyG
NeighborLoader implementation; for truly massive graphs use a disk-backed
graph store / partitioned training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEFAULT_FILENAME = "HI-Small_Trans.csv"
REQUIRED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


def download_kaggle_dataset(dataset_slug: str = KAGGLE_DATASET,
                            download_dir: str = "data/kaggle") -> Path:
    """Download through KaggleHub; Kaggle credentials may be required."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Install kagglehub first: pip install kagglehub. "
            "Kaggle may ask you to authenticate (KAGGLE_USERNAME/KAGGLE_KEY "
            "or a Kaggle API token)."
        ) from exc
    path = Path(kagglehub.dataset_download(dataset_slug, output_dir=download_dir))
    return path


def locate_transaction_csv(path: str | Path,
                           preferred_filename: str = DEFAULT_FILENAME) -> Path:
    """Accept a CSV path or a KaggleHub dataset directory."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    matches = list(path.rglob(preferred_filename))
    if not matches:
        matches = list(path.rglob("*Trans.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No transaction CSV found under {path}. "
            "Expected HI-Small_Trans.csv or another *Trans.csv file."
        )
    # Prefer the requested size/file if multiple files are present.
    matches.sort(key=lambda p: (p.name != preferred_filename, len(str(p))))
    return matches[0]


def _account_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    # Bank is part of the identity: account numbers need not be globally unique.
    return bank.astype(str) + "::" + account.astype(str)


def _scan_metadata(csv_path: Path, chunksize: int):
    currencies, formats = set(), set()
    min_time, max_time = None, None
    rows = 0
    for chunk in pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, chunksize=chunksize):
        for col in ("Payment Currency", "Receiving Currency"):
            currencies.update(chunk[col].dropna().astype(str).unique().tolist())
        formats.update(chunk["Payment Format"].dropna().astype(str).unique().tolist())
        ts = pd.to_datetime(chunk["Timestamp"], errors="coerce", utc=True)
        lo, hi = ts.min(), ts.max()
        if pd.notna(lo):
            min_time = lo if min_time is None else min(min_time, lo)
            max_time = hi if max_time is None else max(max_time, hi)
        rows += len(chunk)
    if not currencies or not formats or min_time is None or max_time is None:
        raise ValueError("Could not read currency, payment-format, or timestamp metadata.")
    return sorted(currencies), sorted(formats), min_time, max_time, rows


def build_graph(csv_path: str | Path, chunksize: int = 250_000,
                make_undirected: bool = True,
                output_path: Optional[str | Path] = None) -> Data:
    """
    Create a PyG Data object.

    Node features:
      * scaled bank identifier (simple baseline encoding)
      * log(1 + mean amount paid) for each currency
      * log(1 + mean amount received) for each currency

    Edge features:
      * normalized timestamp
      * log(1 + amount received), log(1 + amount paid)
      * one-hot receiving currency, payment currency, and payment format

    Note: aggregate node features use all transactions in the selected CSV.
    This is a transductive learning setup and is not a strict future-time
    forecasting evaluation. See notebook discussion before interpreting scores.
    """
    csv_path = locate_transaction_csv(csv_path)
    currencies, formats, min_time, max_time, n_rows = _scan_metadata(csv_path, chunksize)
    currency_to_idx = {v: i for i, v in enumerate(currencies)}
    format_to_idx = {v: i for i, v in enumerate(formats)}
    c_count, f_count = len(currencies), len(formats)

    account_to_id: dict[str, int] = {}
    node_bank: list[float] = []
    paid_sum: list[np.ndarray] = []
    paid_count: list[np.ndarray] = []
    recv_sum: list[np.ndarray] = []
    recv_count: list[np.ndarray] = []
    node_y: list[int] = []

    edge_indices, edge_features = [], []
    bank_values = []

    def get_node_id(key: str, bank_value: float) -> int:
        if key not in account_to_id:
            idx = len(account_to_id)
            account_to_id[key] = idx
            node_bank.append(float(bank_value))
            paid_sum.append(np.zeros(c_count, dtype=np.float64))
            paid_count.append(np.zeros(c_count, dtype=np.float64))
            recv_sum.append(np.zeros(c_count, dtype=np.float64))
            recv_count.append(np.zeros(c_count, dtype=np.float64))
            node_y.append(0)
        return account_to_id[key]

    time_span = max((max_time - min_time).total_seconds(), 1.0)
    for chunk in pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, chunksize=chunksize):
        chunk = chunk.dropna(subset=[
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid",
            "Payment Currency", "Payment Format", "Is Laundering"
        ]).copy()
        if chunk.empty:
            continue

        src_keys = _account_key(chunk["From Bank"], chunk["Account"])
        dst_keys = _account_key(chunk["To Bank"], chunk["Account.1"])
        src_ids = np.empty(len(chunk), dtype=np.int64)
        dst_ids = np.empty(len(chunk), dtype=np.int64)
        labels = chunk["Is Laundering"].astype(int).to_numpy()

        # Account identifiers and node-level aggregates.
        for pos, (src_key, dst_key, src_bank, dst_bank) in enumerate(zip(
            src_keys, dst_keys, chunk["From Bank"], chunk["To Bank"]
        )):
            src = get_node_id(str(src_key), float(src_bank))
            dst = get_node_id(str(dst_key), float(dst_bank))
            src_ids[pos], dst_ids[pos] = src, dst
            if labels[pos] == 1:
                node_y[src] = 1
                node_y[dst] = 1

        paid_amount = pd.to_numeric(chunk["Amount Paid"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        recv_amount = pd.to_numeric(chunk["Amount Received"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        pay_cur = chunk["Payment Currency"].astype(str).map(currency_to_idx).to_numpy()
        recv_cur = chunk["Receiving Currency"].astype(str).map(currency_to_idx).to_numpy()

        for sid, did, pa, ra, pc, rc in zip(src_ids, dst_ids, paid_amount, recv_amount, pay_cur, recv_cur):
            paid_sum[sid][pc] += pa
            paid_count[sid][pc] += 1
            recv_sum[did][rc] += ra
            recv_count[did][rc] += 1

        ts = pd.to_datetime(chunk["Timestamp"], errors="coerce", utc=True)
        seconds = (ts - min_time).dt.total_seconds().fillna(0).to_numpy(dtype=np.float32)
        time_norm = seconds / time_span
        pay_format = chunk["Payment Format"].astype(str).map(format_to_idx).to_numpy()

        # Dense one-hot edge encodings make category semantics explicit.
        n = len(chunk)
        edge_attr = np.zeros((n, 3 + 2 * c_count + f_count), dtype=np.float32)
        edge_attr[:, 0] = time_norm
        edge_attr[:, 1] = np.log1p(paid_amount)
        edge_attr[:, 2] = np.log1p(recv_amount)
        edge_attr[np.arange(n), 3 + c_count + pay_cur] = 1.0
        edge_attr[np.arange(n), 3 + recv_cur] = 1.0
        edge_attr[np.arange(n), 3 + 2 * c_count + pay_format] = 1.0

        edge_indices.append(np.stack([src_ids, dst_ids], axis=0))
        edge_features.append(edge_attr)

    if not account_to_id:
        raise ValueError("No usable transactions were found in the CSV.")

    # Bank values are normalized to keep this baseline feature numerically stable.
    bank_arr = np.asarray(node_bank, dtype=np.float32)
    bank_min, bank_max = float(bank_arr.min()), float(bank_arr.max())
    bank_scaled = (bank_arr - bank_min) / max(bank_max - bank_min, 1.0)

    x = np.zeros((len(account_to_id), 1 + 2 * c_count), dtype=np.float32)
    x[:, 0] = bank_scaled
    for i in range(len(account_to_id)):
        paid_mean = np.divide(paid_sum[i], paid_count[i],
                              out=np.zeros(c_count), where=paid_count[i] > 0)
        recv_mean = np.divide(recv_sum[i], recv_count[i],
                              out=np.zeros(c_count), where=recv_count[i] > 0)
        x[i, 1:1+c_count] = np.log1p(paid_mean)
        x[i, 1+c_count:] = np.log1p(recv_mean)

    edge_index_np = np.concatenate(edge_indices, axis=1)
    edge_attr_np = np.concatenate(edge_features, axis=0)
    y_np = np.asarray(node_y, dtype=np.float32)

    if make_undirected:
        # A transaction can inform both accounts' representations. The two
        # directions share the same transaction attributes.
        edge_index_np = np.concatenate([edge_index_np, edge_index_np[::-1]], axis=1)
        edge_attr_np = np.concatenate([edge_attr_np, edge_attr_np], axis=0)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index_np).long(),
        edge_attr=torch.from_numpy(edge_attr_np).float(),
        y=torch.from_numpy(y_np).float(),
        num_nodes=len(account_to_id),
    )
    data.num_transactions = int(n_rows)
    data.currency_names = currencies
    data.payment_format_names = formats
    data.feature_notes = {
        "node_features": ["scaled_bank_id"]
            + [f"log_mean_paid_{c}" for c in currencies]
            + [f"log_mean_received_{c}" for c in currencies],
        "edge_features": ["normalized_timestamp", "log_amount_paid", "log_amount_received"]
            + [f"receiving_currency={c}" for c in currencies]
            + [f"payment_currency={c}" for c in currencies]
            + [f"payment_format={f}" for f in formats],
        "label_definition": "1 if account is either endpoint of at least one laundering transaction",
    }
    source_path = Path(csv_path).resolve()
    source_stat = source_path.stat()
    data.metadata_json = json.dumps({
        "source_csv": str(source_path),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "transactions_in_csv": int(n_rows),
        "num_accounts": int(data.num_nodes),
        "num_directed_edges_after_option": int(data.edge_index.size(1)),
        "currencies": currencies,
        "payment_formats": formats,
        "make_undirected": make_undirected,
        "timestamp_min_utc": str(min_time),
        "timestamp_max_utc": str(max_time),
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        print(f"Saved graph to {output_path}")
    return data


def load_or_build_graph(csv_path: str | Path, processed_path: str | Path,
                        chunksize: int = 250_000, force_rebuild: bool = False,
                        make_undirected: bool = True) -> Data:
    processed_path = Path(processed_path)
    if processed_path.exists() and not force_rebuild:
        try:
            try:
                cached = torch.load(processed_path, map_location="cpu", weights_only=False)
            except TypeError:  # older PyTorch
                cached = torch.load(processed_path, map_location="cpu")
            source_path = Path(locate_transaction_csv(csv_path)).resolve()
            source_stat = source_path.stat()
            metadata = json.loads(getattr(cached, "metadata_json", "{}"))
            cache_matches = (
                metadata.get("source_csv") == str(source_path)
                and metadata.get("source_size_bytes") == source_stat.st_size
                and metadata.get("source_mtime_ns") == source_stat.st_mtime_ns
                and metadata.get("make_undirected") == make_undirected
            )
            if cache_matches:
                return cached
            print("Cached graph does not match the selected CSV or edge direction; rebuilding.")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, AttributeError):
            print("Could not validate the cached graph; rebuilding.")
    return build_graph(csv_path, chunksize, make_undirected, processed_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an AML transaction graph.")
    parser.add_argument("--csv", type=str, default=None, help="CSV path or KaggleHub folder.")
    parser.add_argument("--download", action="store_true", help="Download the Kaggle dataset first.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--output", type=str, default="data/processed/aml_graph.pt")
    parser.add_argument("--directed", action="store_true", help="Do not add reverse edges.")
    args = parser.parse_args()

    source = args.csv
    if args.download or source is None:
        source = download_kaggle_dataset()
    csv_file = locate_transaction_csv(source, args.filename)
    build_graph(csv_file, chunksize=args.chunksize,
                make_undirected=not args.directed, output_path=args.output)

# Graph Attention Networks for Anti-Money Laundering (AML)

Educational, account-level node-classification project using the IBM AML transaction dataset on Kaggle.

## Project layout

- `dataset.py` — KaggleHub download, chunked CSV preprocessing, graph construction.
- `model.py` — two-layer, edge-aware Graph Attention Network.
- `train.py` — stratified node split, neighbor sampling, weighted loss, validation and test metrics.
- `anti-money-laundering-detection-with-gnn.ipynb` — guided notebook with explanations and hands-on model steps.

## 1. Environment

Use Python 3.10+ (a clean virtual environment is recommended).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Install a PyTorch build suitable for your CUDA version from the official PyTorch selector if you want GPU support. `NeighborLoader` also needs either `pyg-lib` or `torch-sparse`; these wheels must match your installed PyTorch and CUDA versions. Follow the matching installation command at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html before running training. The script checks for this sampling backend before downloading the dataset and prints the setup link if it is missing.

## 2. Kaggle access and download

The project downloads the selected CSV directly into `data/raw/` through KaggleHub instead of downloading every dataset variant. The default is `HI-Small_Trans.csv`, recommended for a first run.

```bash
pip install kagglehub
```

Authenticate Kaggle as described by KaggleHub / Kaggle API. Common approaches are setting `KAGGLE_USERNAME` and `KAGGLE_KEY`, or configuring a Kaggle API token in your environment. Never commit credentials or `kaggle.json` to GitHub.

Run training with automatic download (defaults to `HI-Small_Trans.csv`). To keep Codespaces memory use bounded, training defaults to a deterministic, uniform sample of 100,000 transactions drawn across the full CSV. This is a quick baseline on a subset, not a full-dataset evaluation.

```bash
python train.py
```

Or point to an already downloaded folder/CSV:

```bash
python train.py --csv "path/to/kaggle/dataset/folder"
```

The default run downloads only `HI-Small_Trans.csv`. Use `--filename` to download a different file; larger variants require substantially more disk space and RAM. For example:

```bash
python train.py --filename HI-Large_Trans.csv --rebuild
```

The graph is cached at `data/processed/aml_graph.pt`; model and metrics are written under `artifacts/`.

## 3. What is the graph?

- **Node:** one account, uniquely identified by `(Bank, Account)`.
- **Directed edge:** one transaction from the sending account to the receiving account.
- **Node label:** 1 if the account was either endpoint of at least one transaction marked `Is Laundering=1`; otherwise 0.
- **Node features:** bank identifier plus log-transformed mean paid/received amounts by currency.
- **Edge features:** normalized time, log-transformed paid/received amounts, and one-hot currency/payment-format categories.

By default each transaction is represented in both directions so each endpoint can aggregate neighborhood information. Use `--directed` to retain only sender-to-receiver edges.

## 4. Model intuition

For each edge, a GAT learns an attention coefficient that represents how useful that neighbor is to the target node. In this implementation, the attention score can also use transaction edge attributes (`edge_dim`): amount, time, currencies and payment format.

Layer 1 has `heads=4`; the four attention-head outputs are concatenated, so the embedding width is `hidden_channels * heads`. Layer 2 produces one hidden vector per account. A linear classifier returns one logit; sigmoid converts it to a probability at inference time.

`BCEWithLogitsLoss` is used rather than applying sigmoid before binary cross entropy. A positive-class weight helps with the severe class imbalance common in AML.

## 5. Important evaluation caveats

This is a learning baseline, not a production AML system.

1. The node label is an **account-level proxy** derived from transaction labels. It does not prove that an account is criminal or that every transaction involving it is illicit.
2. The split is random and transductive. All graph edges remain visible to message passing; only supervised seed nodes are split into train/validation/test. Node aggregates are computed over the selected CSV, including transactions associated with held-out nodes. This can leak future or held-out-period information for a real deployment scenario.
3. For a realistic prospective AML evaluation, create a chronological split, build features using only transactions available before the prediction cutoff, and evaluate on later transactions/accounts. Also consider transaction-level prediction and alert-level review.
4. Accuracy alone is misleading when positives are rare. Inspect PR-AUC (average precision), recall, precision, F1, and the confusion matrix. Choose the alert threshold using validation data and operational capacity.
5. `HI-Large` can require substantial RAM and GPU memory. Chunked preprocessing avoids loading the raw CSV as one DataFrame, but the final graph and sampled subgraphs still occupy memory. Reduce `--batch-size`, `--neighbors`, or use `HI-Small` to learn the architecture first.

## 6. Train

```bash
python train.py --epochs 100 --batch-size 256 --lr 0.0001 --neighbors 8 --chunksize 50000
```

If memory is tight:

```bash
python train.py --batch-size 256 --neighbors 8 --chunksize 50000
```

The script uses `NeighborLoader`; depending on your PyG build, neighborhood sampling may require `pyg-lib` or `torch-sparse`. Follow the installation guidance in the PyG documentation for your environment. Increase `--max-transactions` for a larger uniform sample, or pass `--max-transactions 0` for the full CSV if your machine has enough memory and time. Sampling is by transaction row and preserves the full file's class prevalence only in expectation; metrics on a small sample are unstable.

## References

- Original dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- PyTorch Geometric GATConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
- Weber et al. (2019), *Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics*: https://arxiv.org/abs/1908.02591
- Johannessen & Jullum (2023), *Finding Money Launderers Using Heterogeneous Graph Neural Networks*: https://arxiv.org/abs/2307.13499

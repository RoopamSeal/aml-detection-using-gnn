# AML Detection with a Simple Graph Neural Network

A small, beginner-friendly Graph Attention Network (GAT) baseline for the IBM Transactions for Anti-Money Laundering dataset.

## Project structure

```text
.
├── dataset.py       # Download data and convert transactions into a graph
├── model.py         # Two-layer edge-aware GAT
├── train.py         # Train, validate, test, and save results
├── requirements.txt
├── aml_gnn.ipynb    # Short walkthrough notebook
├── data/            # Generated data (not committed)
└── artifacts/       # Model, metrics, and plots (not committed)
```

## Run in GitHub Codespaces

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch Geometric's basic `GATConv` works without the optional `pyg-lib` / `torch-sparse` neighbor-sampling extensions. The project intentionally uses full-batch training on a small graph.

### 2. Configure Kaggle access

Accept the dataset's terms on Kaggle. Create a Kaggle API token and configure it as a Codespaces secret/environment variable, or use KaggleHub's supported authentication flow. Do not commit `kaggle.json`, API tokens, or credentials.

### 3. Train

```bash
python train.py
```

By default the script uses up to 25,000 rows from `HI-Small_Trans.csv`, trains for up to 20 epochs, and stops early if validation PR-AUC does not improve.

To use a local CSV:

```bash
python train.py --csv data/HI-Small_Trans.csv
```

To change the sample size or training duration:

```bash
python train.py --max-transactions 40000 --epochs 30
```

### 4. Outputs

- `artifacts/best_gat.pt` — best model checkpoint selected using validation PR-AUC
- `artifacts/metrics.json` — validation history and held-out test metrics
- `artifacts/training_loss.png`
- `artifacts/validation_pr_auc.png`

## Graph and model

- **Node:** a bank/account pair (`From Bank` + `Account`, or `To Bank` + `Account.1`).
- **Edge:** a transaction from sender to receiver. Reverse edges are added for message passing.
- **Node features:** normalized bank ID, log mean amount paid, log mean amount received, log sent count, and log received count.
- **Edge features:** normalized timestamp and log amounts paid/received.
- **Target:** an account is labelled positive if it is an endpoint of at least one sampled transaction with `Is Laundering = 1`.
- **Model:** two GAT layers, followed by a linear binary classifier.

## Important limitations

This is a compact learning baseline, not a production AML system. The target is an account-level proxy derived from transaction labels. Features and graph structure are built from the same sampled time period, so the experiment is transductive and is not a leakage-free future forecasting evaluation. A stronger study should construct features and graph edges only from transactions before a time cutoff, then evaluate on later transactions/accounts. Also compare against non-graph baselines and report threshold-dependent metrics alongside PR-AUC.

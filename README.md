# Zero-Day Fraud Detection in Dynamic Heterogeneous Transaction Networks using Hybrid GNN + Graph Autoencoder

A Graph Machine Learning based fraud detection framework designed for detecting both **known fraud patterns** and **zero-day (previously unseen) fraudulent activities** using a hybrid deep learning approach combining:

- Variational Graph Autoencoder (VGAE)
- Relational Graph Convolution Network (RGCN)
- Fusion Neural Network
- Dynamic graph-based transaction modeling

The project models financial transactions as graph structures and combines:

- **Unsupervised anomaly learning** using VGAE
- **Supervised graph learning** using RGCN
- **Fusion of graph embeddings** for final fraud prediction

---

# Project Motivation

Traditional fraud detection systems mainly rely on:

- static tabular datasets
- supervised learning
- large labeled datasets

These approaches struggle to detect:

- unseen fraud patterns
- evolving attacker behavior
- zero-day fraud scenarios

This project addresses those limitations by leveraging **Graph Machine Learning** techniques capable of learning:

- transaction relationships
- structural dependencies
- behavioral patterns
- graph anomalies

---

# Key Features

- Graph-based transaction modeling
- Zero-day fraud detection using VGAE
- Relational fraud learning using RGCN
- Hybrid fusion architecture
- Dynamic transaction network representation
- Transaction Network Graph visualization
- Fraud prediction pipeline
- Performance evaluation & visualization

---

# System Architecture

```text
Datasets
   ↓

Preprocessing
   ↓

Graph Construction
   ↓

Dynamic Transaction Network Graph
   ↓

VGAE
(Zero-day anomaly detection)
   ↓

RGCN
(Relational fraud learning)
   ↓

Fusion Layer
   ↓

Final Fraud Prediction
   ↓

Evaluation & Visualization
```

---

# Datasets Used

The project uses multiple datasets to simulate realistic financial transaction environments.

---

## 1. PaySim Dataset

Synthetic mobile money transaction dataset.

### Features

- transaction amount
- transaction type
- sender ID
- receiver ID
- timestamp
- fraud label

### Expected file

```text
datasets/
    PS_20174392719_1491204439457_log.csv
```

---

## 2. Elliptic Bitcoin Dataset

Bitcoin transaction graph dataset.

### Features

- transaction IDs
- transaction classes
- temporal information
- transaction edges
- transaction connectivity

### Expected files

```text
datasets/
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
```

---

# Folder Structure

```text
Hybrid-Fraud-Detection/

│
├── datasets/
│   │
│   ├── PS_20174392719_1491204439457_log.csv
│   │
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   │
│   ├── preprocessing.py
│   ├── graph_builder.py
│   ├── vgae_model.py
│   ├── rgcn_model.py
│   ├── fusion_model.py
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   └── transaction_network.py
│
├── outputs/
│   │
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── transaction_network.png
│   ├── vgae_loss.png
│   ├── rgcn_loss.png
│   └── fusion_loss.png
│
├── hybrid_model.pt
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/Hybrid-Fraud-Detection.git
```

Move into project folder:

```bash
cd Hybrid-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Libraries

Main libraries used:

```text
pandas
numpy
scikit-learn
matplotlib
networkx
torch
torch-geometric
```

Manual installation:

```bash
pip install pandas numpy scikit-learn matplotlib networkx torch torch-geometric
```

---

# Dataset Setup

Create:

```text
datasets/
```

Place dataset files exactly as:

```text
datasets/

│
├── PS_20174392719_1491204439457_log.csv

├── elliptic_txs_features.csv

├── elliptic_txs_classes.csv

└── elliptic_txs_edgelist.csv
```

---

# Running the Project

---

## Step 1 — Train Hybrid Model

Run:

```bash
python src/train.py
```

This performs:

- dataset loading
- preprocessing
- graph construction
- transaction network generation
- VGAE training
- RGCN training
- fusion model training
- evaluation
- model saving

### Generated files

```text
hybrid_model.pt
```

### Saved intermediate outputs

```text
results_true.npy

results_pred.npy

results_prob.npy

vgae_losses.npy

rgcn_losses.npy

fusion_losses.npy
```

---

## Step 2 — Run Prediction

After training:

```bash
python src/predict.py
```

This performs:

- loading trained model
- graph reconstruction
- fraud prediction

### Example output

```text
First 20 predictions:

tensor(
[0,1,0,0,1...]
)
```

Where:

```text
0 → Normal Transaction

1 → Fraud Transaction
```

---

## Step 3 — Generate Visualizations

Run:

```bash
python src/visualize.py
```

Generated visual outputs:

```text
confusion_matrix.png

roc_curve.png

transaction_network.png

vgae_loss.png

rgcn_loss.png

fusion_loss.png
```

---

# Transaction Network Graph

The project represents financial transactions as a **directed graph network**.

### Graph Representation

```text
Nodes  → Accounts / Users / Transactions

Edges  → Transaction relationships
```

### Purpose

The Transaction Network Graph helps visualize:

- suspicious transaction patterns
- fraud clusters
- abnormal node connectivity
- graph structure learned by GNNs

### Fraud Visualization

- Normal nodes → blue
- Fraud nodes → red

This graph demonstrates the core Graph Machine Learning concept used in the project.

---

# Evaluation Metrics

The project evaluates performance using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

---

# Example Output

```text
Classification Report:

precision    recall   f1-score

Normal:

0.98

Fraud:

0.20


ROC-AUC:

0.737
```

---

# Generated Visualizations

---

## 1. Transaction Network Graph

Shows:

- transaction relationships
- connected accounts
- fraud node patterns
- graph topology

---

## 2. ROC Curve

Shows:

- True Positive Rate
- False Positive Rate
- model discrimination capability

---

## 3. Confusion Matrix

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

## 4. Loss Curves

Training convergence for:

- VGAE
- RGCN
- Fusion Model

---

# Core Models

---

## VGAE (Variational Graph Autoencoder)

Used for:

- graph anomaly detection
- latent graph representation learning
- zero-day fraud identification

---

## RGCN (Relational Graph Convolution Network)

Used for:

- supervised fraud learning
- relational message passing
- neighborhood aggregation

---

## Fusion Model

Combines:

```text
VGAE embeddings

+

RGCN embeddings
```

to generate:

```text
Final Fraud Prediction
```

---

# Future Improvements

Possible future extensions:

- temporal graph snapshots
- dynamic graph learning
- Graph Attention Networks (GAT)
- explainable AI methods
- real-time fraud detection
- fully heterogeneous graph modeling

Additional node types:

- users
- merchants
- devices
- IP addresses
- locations

---

# Research Contribution

This project contributes toward:

- zero-day fraud detection
- graph-based anomaly detection
- hybrid GNN architectures
- dynamic transaction network analysis

using Graph Machine Learning techniques.

---

# License

This project is intended for academic and research purposes.

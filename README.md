````markdown
# Zero-Day Fraud Detection in Dynamic Heterogeneous Transaction Networks using Hybrid GNN + Graph Autoencoder

A Graph Machine Learning based fraud detection framework designed for detecting both **known fraud patterns** and **zero-day (previously unseen) fraudulent activities** using a hybrid approach combining:

- Variational Graph Autoencoder (VGAE)
- Relational Graph Convolution Network (RGCN)
- Fusion Neural Network
- Dynamic graph-based transaction modeling

The project models financial transactions as graph structures and combines:

- **Unsupervised anomaly learning** using VGAE
- **Supervised graph learning** using RGCN
- **Fusion of both embeddings** for final fraud prediction

---

## Project Motivation

Traditional fraud detection systems rely heavily on:

- Static tabular datasets
- Supervised learning
- Large labeled datasets

These approaches struggle to detect:

- unseen fraud patterns
- evolving attacker behavior
- zero-day fraud scenarios

This project addresses those limitations by leveraging Graph Machine Learning techniques that learn both:

- structural relationships
- transaction behavior patterns

---

# System Architecture

```text
Datasets
   ↓

Preprocessing
   ↓

Graph Construction
   ↓

Heterogeneous Transaction Graph
   ↓

VGAE
(Zero-day anomaly detection)
   ↓

RGCN
(Known fraud learning)
   ↓

Fusion Layer
   ↓

Final Fraud Prediction
   ↓

Evaluation
```

---

# Datasets Used

The project uses multiple datasets to simulate real-world transaction environments.

## 1. PaySim Dataset

Synthetic mobile money transaction dataset containing:

Features:

- transaction amount
- transaction type
- sender ID
- receiver ID
- timestamp
- fraud label

Expected file:

```text
datasets/
    PS_20174392719_1491204439457_log.csv
```

---

## 2. Elliptic Bitcoin Dataset

Bitcoin transaction graph dataset containing:

Features:

- transaction IDs
- transaction classes
- temporal information
- transaction connections

Expected files:

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
│   └── visualize.py
│
├── requirements.txt
│
├── hybrid_model.pt
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
torch
torch-geometric
```

Install manually if needed:

```bash
pip install pandas numpy scikit-learn matplotlib torch torch-geometric
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

Note:

Dataset files are ignored using `.gitignore` because:

- datasets are large
- GitHub has size limitations

---

# Running the Project

## Step 1: Train Hybrid Model

Run:

```bash
python src/train.py
```

This performs:

- dataset loading
- preprocessing
- graph creation
- VGAE training
- RGCN training
- fusion model training
- evaluation
- model saving

Output:

```text
hybrid_model.pt
```

Saved intermediate files:

```text
results_true.npy

results_pred.npy

results_prob.npy

vgae_losses.npy

rgcn_losses.npy

fusion_losses.npy
```

---

## Step 2: Run Prediction

After training:

```bash
python src/predict.py
```

This performs:

- loading saved model
- graph generation
- fraud prediction

Example output:

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

## Step 3: Generate Visualization Results

Run:

```bash
python src/visualize.py
```

Generated images:

```text
confusion_matrix.png

roc_curve.png

vgae_loss.png

rgcn_loss.png

fusion_loss.png
```

---

# Evaluation Metrics

The project evaluates performance using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
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

The project generates:

### Confusion Matrix

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC Curve

Shows:

- True Positive Rate
- False Positive Rate

### Training Loss Curves

For:

- VGAE
- RGCN
- Fusion Model

---

# Core Models

## VGAE

Used for:

- anomaly detection
- latent graph representation
- zero-day fraud identification

---

## RGCN

Used for:

- supervised fraud learning
- neighborhood information aggregation

---

## Fusion Model

Combines:

```text
VGAE embeddings

+

RGCN embeddings
```

to produce:

```text
Final Fraud Prediction
```

---

# Future Improvements

Possible extensions:

- Temporal graph snapshots
- Dynamic graph learning
- Graph Attention Networks (GAT)
- Explainable AI methods
- Real-time fraud detection
- Additional heterogeneous node types:

    - users
    - merchants
    - devices
    - IP addresses

---

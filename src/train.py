# =====================================================
# src/train.py
# =====================================================

import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from graph_builder import create_graph
from preprocessing import preprocess_paysim, preprocess_elliptic
from rgcn_model import RGCN
from fusion_model import HybridFusionModel
from vgae_model import Encoder

from torch_geometric.nn import VGAE


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# =====================================================
# LOAD + PREPROCESS
# =====================================================

print("\nLoading datasets...")

pay = preprocess_paysim(
    str(DATA_DIR / 'PS_20174392719_1491204439457_log.csv' / 'PS_20174392719_1491204439457_log.csv')
)

elliptic = preprocess_elliptic(
    str(DATA_DIR / 'elliptic_txs_features.csv' / 'elliptic_txs_features.csv'),
    str(DATA_DIR / 'elliptic_txs_classes.csv' / 'elliptic_txs_classes.csv'),
    str(DATA_DIR / 'elliptic_txs_edgelist.csv' / 'elliptic_txs_edgelist.csv'),
)


# =====================================================
# COMBINE DATASETS
# =====================================================

combined = pd.concat([

    pay,
    elliptic

])


print("Combined shape:")

print(combined.shape)


# =====================================================
# CREATE GRAPH
# =====================================================

graph = create_graph(combined)
graph = graph.to(DEVICE)

x = graph.x

y = graph.y
edge_index = graph.edge_index

# Normalize node features
scaler = StandardScaler()
x = torch.tensor(
    scaler.fit_transform(x.cpu().numpy()),
    dtype=torch.float,
    device=DEVICE,
)

# Split nodes for training, validation, and testing
num_nodes = y.size(0)
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(
    indices,
    stratify=y.cpu().numpy(),
    test_size=0.2,
    random_state=42,
)
train_idx, val_idx = train_test_split(
    train_idx,
    stratify=y[train_idx].cpu().numpy(),
    test_size=0.1,
    random_state=42,
)

train_idx = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
val_idx = torch.tensor(val_idx, dtype=torch.long, device=DEVICE)
test_idx = torch.tensor(test_idx, dtype=torch.long, device=DEVICE)

train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

edge_type = torch.zeros(
    edge_index.shape[1],
    dtype=torch.long,
    device=DEVICE,
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y.cpu().numpy()),
    y=y.cpu().numpy(),
)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)
# Use a moderate focal alpha to balance classes without extreme all-positive prediction
alpha = torch.tensor([0.3, 0.7], dtype=torch.float, device=DEVICE)

loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

print("\nGraph created")
print(graph)
print(f"Train/Val/Test nodes: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")


# =====================================================
# TRAIN VGAE
# =====================================================

print("\nTraining VGAE...")


encoder = Encoder(

    x.shape[1],

    16

)

vgae_model = VGAE(
    encoder
).to(DEVICE)

vgae_losses = []

optimizer = torch.optim.Adam(
    vgae_model.parameters(),
    lr=0.01
)

for epoch in range(50):

    vgae_model.train()

    optimizer.zero_grad()

    z = vgae_model.encode(

        x,

        edge_index

    )

    loss = vgae_model.recon_loss(

        z,

        edge_index

    )

    loss.backward()
    optimizer.step()
    vgae_losses.append(loss.item())

    if epoch%10==0:
        print(
            f"Epoch {epoch}: {loss.item():.4f}"
        )


print("\nVGAE complete")


# =====================================================
# GENERATE VGAE EMBEDDINGS
# =====================================================

vgae_model.eval()

with torch.no_grad():

    vgae_embeddings = vgae_model.encode(

        x,

        edge_index

    )


print(

    "VGAE embedding shape:",

    vgae_embeddings.shape

)


# =====================================================
# TRAIN RGCN
# =====================================================

print("\nTraining RGCN...")

rgcn_model = RGCN(
    x.shape[1]
).to(DEVICE)

rgcn_losses = []
optimizer = torch.optim.Adam(
    rgcn_model.parameters(),
    lr=0.01
)

for epoch in range(75):
    rgcn_model.train()
    optimizer.zero_grad()

    out = rgcn_model(
        x,
        edge_index,
        edge_type
    )

    loss = loss_fn(
        out[train_mask],
        y[train_mask]
    )

    loss.backward()
    optimizer.step()
    rgcn_losses.append(loss.item())

    if epoch % 10 == 0:
        with torch.no_grad():
            val_out = rgcn_model(x, edge_index, edge_type)
            val_prob = torch.softmax(val_out[val_mask], dim=1)[:, 1]
            val_pred = val_out[val_mask].argmax(dim=1)
            val_acc = accuracy_score(
                y[val_mask].cpu(),
                val_pred.cpu()
            )
            val_f1 = f1_score(
                y[val_mask].cpu(),
                val_pred.cpu(),
                zero_division=0
            )
            val_auc = roc_auc_score(
                y[val_mask].cpu(),
                val_prob.cpu()
            )
        print(
            f"Epoch {epoch}: {loss.item():.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_auc: {val_auc:.4f}"
        )

print("\nRGCN complete")


# =====================================================
# RGCN EMBEDDINGS
# =====================================================

rgcn_model.eval()

with torch.no_grad():

    rgcn_embeddings = rgcn_model(

        x,

        edge_index,

        edge_type

    )


print(

    "RGCN embedding shape:",

    rgcn_embeddings.shape

)


# =====================================================
# HYBRID FUSION
# =====================================================

print("\nTraining Fusion Model...")

fusion_model = HybridFusionModel(
    vgae_dim=vgae_embeddings.shape[1],
    rgcn_dim=rgcn_embeddings.shape[1]
).to(DEVICE)

fusion_losses = []
optimizer = torch.optim.Adam(
    fusion_model.parameters(),
    lr=0.001
)

for epoch in range(100):
    fusion_model.train()
    optimizer.zero_grad()

    out = fusion_model(
        vgae_embeddings,
        rgcn_embeddings
    )

    loss = loss_fn(
        out[train_mask],
        y[train_mask]
    )

    loss.backward()
    optimizer.step()
    fusion_losses.append(loss.item())

    if epoch % 10 == 0:
        with torch.no_grad():
            val_out = fusion_model(vgae_embeddings, rgcn_embeddings)
            val_prob = torch.softmax(val_out[val_mask], dim=1)[:, 1]
            val_pred = val_out[val_mask].argmax(dim=1)
            val_acc = accuracy_score(
                y[val_mask].cpu(),
                val_pred.cpu()
            )
            val_f1 = f1_score(
                y[val_mask].cpu(),
                val_pred.cpu(),
                zero_division=0
            )
            val_auc = roc_auc_score(
                y[val_mask].cpu(),
                val_prob.cpu()
            )
        print(
            f"Epoch {epoch}: {loss.item():.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_auc: {val_auc:.4f}"
        )


# =====================================================
# EVALUATION
# =====================================================

fusion_model.eval()

with torch.no_grad():

    pred=fusion_model(

        vgae_embeddings,

        rgcn_embeddings

    )


test_pred = pred[test_mask].argmax(
    dim=1
)

test_labels = y[test_mask]


print("\nClassification Report:\n")

print(

    classification_report(

        test_labels.cpu(),

        test_pred.cpu()

    )

)


test_accuracy = accuracy_score(
    test_labels.cpu(),
    test_pred.cpu()
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

prob=torch.softmax(

    pred,

    dim=1

)[:,1]


auc=roc_auc_score(

    test_labels.cpu(),

    prob[test_mask].cpu()

)


print(

    "\nROC-AUC:",

    auc

)


# =====================================================
# SAVE MODEL
# =====================================================

torch.save(

    fusion_model.state_dict(),

    'hybrid_model.pt'

)


print(

"\nSaved model: hybrid_model.pt"

)

# =====================================================
# SAVE RESULTS FOR VISUALIZATION
# =====================================================

import numpy as np

np.save(

    "results_true.npy",

    y[test_idx].cpu().numpy()

)

np.save(

    "results_pred.npy",

    test_pred.cpu().numpy()

)

np.save(

    "results_prob.npy",

    prob[test_mask].cpu().numpy()

)


np.save(

    "vgae_losses.npy",

    np.array(vgae_losses)

)

np.save(

    "rgcn_losses.npy",

    np.array(rgcn_losses)

)

np.save(

    "fusion_losses.npy",

    np.array(fusion_losses)

)

print("\nResults saved")
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_paysim, preprocess_elliptic
from graph_builder import create_graph
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datasets"

# Recreate combined dataframe and graph to obtain y and test indices
pay = preprocess_paysim(
    str(DATA_DIR / 'PS_20174392719_1491204439457_log.csv' / 'PS_20174392719_1491204439457_log.csv')
)
elliptic = preprocess_elliptic(
    str(DATA_DIR / 'elliptic_txs_features.csv' / 'elliptic_txs_features.csv'),
    str(DATA_DIR / 'elliptic_txs_classes.csv' / 'elliptic_txs_classes.csv'),
    str(DATA_DIR / 'elliptic_txs_edgelist.csv' / 'elliptic_txs_edgelist.csv'),
)
combined = pd.concat([pay, elliptic])

# build graph to get y
graph = create_graph(combined)
y = graph.y

# deterministic split to match train.py
indices = np.arange(y.size(0))
train_idx, test_idx = train_test_split(indices, stratify=y.cpu().numpy(), test_size=0.2, random_state=42)
# create test mask
test_mask = np.zeros(y.size(0), dtype=bool)
test_mask[test_idx] = True

src_dir = Path(__file__).parent
# save y in test_mask order inside src
out_path = src_dir / 'results_true.npy'
np.save(out_path, y[test_mask].cpu().numpy())
print('Saved', out_path, 'shape', np.load(out_path).shape)

# quick check: compute roc_auc with existing results_prob.npy in src
from sklearn.metrics import roc_auc_score
prob = np.load(src_dir / 'results_prob.npy')
true = np.load(out_path)
print('roc_auc:', roc_auc_score(true, prob))

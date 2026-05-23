# =====================================================
# predict.py
# =====================================================

import torch
import pandas as pd
from pathlib import Path

from preprocessing import preprocess_paysim, preprocess_elliptic
from graph_builder import create_graph
from vgae_model import Encoder
from rgcn_model import RGCN
from fusion_model import HybridFusionModel
from torch_geometric.nn import VGAE

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datasets"
DEVICE = torch.device('cpu')


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model=HybridFusionModel(

    vgae_dim=16,

    rgcn_dim=2

)

model.load_state_dict(

    torch.load(

        'hybrid_model.pt',

        map_location=DEVICE

    )

)

model.eval()

print("Model loaded successfully")


# =====================================================
# LOAD SAMPLE DATA
# =====================================================

pay = preprocess_paysim(
    str(DATA_DIR / 'PS_20174392719_1491204439457_log.csv' / 'PS_20174392719_1491204439457_log.csv')
)

elliptic = preprocess_elliptic(
    str(DATA_DIR / 'elliptic_txs_features.csv' / 'elliptic_txs_features.csv'),
    str(DATA_DIR / 'elliptic_txs_classes.csv' / 'elliptic_txs_classes.csv'),
    str(DATA_DIR / 'elliptic_txs_edgelist.csv' / 'elliptic_txs_edgelist.csv'),
)


combined=pd.concat([

pay,
elliptic

])


graph = create_graph(
    combined
)

graph = graph.to(DEVICE)

x = graph.x
edge_index = graph.edge_index


edge_type = torch.zeros(
    edge_index.shape[1],
    dtype=torch.long
)


# =====================================================
# RECREATE EMBEDDINGS
# =====================================================

vgae_model=VGAE(

Encoder(

x.shape[1],

16

)

)


with torch.no_grad():

    vgae_embeddings=vgae_model.encode(

        x,

        edge_index

    )


rgcn_model=RGCN(

x.shape[1]

)


with torch.no_grad():

    rgcn_embeddings=rgcn_model(

        x,

        edge_index,

        edge_type

    )


# =====================================================
# FRAUD PREDICTION
# =====================================================

with torch.no_grad():

    pred=model(

        vgae_embeddings,

        rgcn_embeddings

    )


prediction=pred.argmax(
    dim=1
)


print("\nFirst 20 predictions:\n")

print(

prediction[:20]

)
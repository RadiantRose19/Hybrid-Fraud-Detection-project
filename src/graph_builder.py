# src/graph_builder.py

import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd


def create_graph(df):

    graph = Data()

    nodes = pd.concat([
        df['source'],
        df['target']
    ]).unique()

    mapping = {
        n: i
        for i, n in enumerate(nodes)
    }

    df = df.copy()
    df['source_id'] = df['source'].map(mapping)
    df['target_id'] = df['target'].map(mapping)

    edge_index = torch.tensor(
        np.vstack([df['source_id'].values, df['target_id'].values]),
        dtype=torch.long,
    )

    node_df = pd.concat([
        df[['source_id', 'amount', 'timestamp', 'label']],
        df[['target_id', 'amount', 'timestamp', 'label']].rename(
            columns={'target_id': 'source_id'}
        )
    ], ignore_index=True)

    node_features = node_df.groupby('source_id').agg({
        'amount': 'mean',
        'timestamp': 'mean',
        'label': 'max'
    }).sort_index()

    x = torch.tensor(
        node_features[['amount', 'timestamp']].values,
        dtype=torch.float
    )

    y = torch.tensor(
        node_features['label'].values,
        dtype=torch.long
    )

    graph.x = x
    graph.y = y
    graph.edge_index = edge_index

    return graph
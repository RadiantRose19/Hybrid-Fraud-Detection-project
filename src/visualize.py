# =====================================================
# visualize.py
# =====================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    nx = None

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# =====================================================
# LOAD SAVED RESULTS
# =====================================================

y_true=np.load(
    "results_true.npy"
)

y_pred=np.load(
    "results_pred.npy"
)

y_prob=np.load(
    "results_prob.npy"
)

vgae_losses=np.load(
    "vgae_losses.npy"
)

rgcn_losses=np.load(
    "rgcn_losses.npy"
)

fusion_losses=np.load(
    "fusion_losses.npy"
)


ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


def load_transaction_network(root_dir):
    if nx is None:
        raise ImportError(
            "networkx is required for transaction network visualization. "
            "Install it with: pip install networkx"
        )

    class_path = os.path.join(
        root_dir,
        "datasets",
        "elliptic_txs_classes.csv",
        "elliptic_txs_classes.csv"
    )
    edge_path = os.path.join(
        root_dir,
        "datasets",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_edgelist.csv"
    )

    classes = pd.read_csv(class_path)
    edges = pd.read_csv(edge_path)

    classes["is_fraud"] = (
        classes["class"] == "1"
    ).astype(int)

    fraud_map = classes.set_index("txId")["is_fraud"].to_dict()

    graph = nx.Graph()
    graph.add_edges_from(
        edges[['txId1', 'txId2']].itertuples(index=False, name=None)
    )

    nx.set_node_attributes(graph, fraud_map, "is_fraud")

    for node in graph.nodes():
        if "is_fraud" not in graph.nodes[node]:
            graph.nodes[node]["is_fraud"] = 0

    return graph


def build_transaction_subgraph(graph, max_nodes=500, max_fraud=60):
    fraud_nodes = sorted(
        [
            n for n, data in graph.nodes(data=True)
            if data.get("is_fraud", 0) == 1
        ],
        key=lambda n: graph.degree(n),
        reverse=True
    )

    if not fraud_nodes:
        return graph

    selected = set(fraud_nodes[: min(max_fraud, len(fraud_nodes), max_nodes // 5)])
    if not selected:
        selected = set(fraud_nodes[:1])

    neighbors = set()
    for node in selected:
        neighbors.update(graph.neighbors(node))

    normal_neighbors = sorted(
        [n for n in neighbors if graph.nodes[n].get("is_fraud", 0) == 0],
        key=lambda n: graph.degree(n),
        reverse=True
    )
    selected.update(normal_neighbors[: max_nodes - len(selected)])

    if len(selected) < max_nodes:
        other_neighbors = sorted(
            neighbors - selected,
            key=lambda n: graph.degree(n),
            reverse=True
        )
        selected.update(other_neighbors[: max_nodes - len(selected)])

    if len(selected) > max_nodes:
        selected = set(list(selected)[:max_nodes])

    subgraph = graph.subgraph(selected).copy()

    if not any(
        data.get("is_fraud", 0) == 0
        for _, data in subgraph.nodes(data=True)
    ):
        normal_candidates = sorted(
            [
                n for n, data in graph.nodes(data=True)
                if data.get("is_fraud", 0) == 0
            ],
            key=lambda n: graph.degree(n),
            reverse=True
        )
        add_count = min(max_nodes - len(selected), len(normal_candidates))
        selected.update(normal_candidates[:add_count])
        subgraph = graph.subgraph(selected).copy()

    return subgraph


def plot_transaction_network(graph):
    graph = build_transaction_subgraph(graph, max_nodes=500)

    colors = [
        "#FF4136" if data.get("is_fraud", 0) == 1 else "#87CEEB"
        for _, data in graph.nodes(data=True)
    ]
    sizes = [
        160 if data.get("is_fraud", 0) == 1 else 40
        for _, data in graph.nodes(data=True)
    ]

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(
        graph,
        seed=42,
        k=0.12,
        iterations=100
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#909090",
        alpha=0.25,
        width=0.5
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=colors,
        node_size=sizes,
        edgecolors="#222222",
        linewidths=0.4,
        alpha=0.85
    )

    plt.title(
        "Transaction Network Graph: Fraud Nodes Highlighted"
    )
    plt.axis("off")

    plt.scatter([], [], c="#87CEEB", label="Normal Node", s=40)
    plt.scatter([], [], c="#FF4136", label="Fraud Node", s=160)
    plt.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("transaction_network.png")
    plt.show()


print("Results loaded")


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm=confusion_matrix(

    y_true,

    y_pred

)

disp=ConfusionMatrixDisplay(

    confusion_matrix=cm

)

disp.plot()

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# =====================================================
# ROC CURVE
# =====================================================

fpr,tpr,_=roc_curve(

    y_true,

    y_prob

)

roc_auc=auc(

    fpr,

    tpr

)

plt.figure()

plt.plot(

    fpr,

    tpr,

    label=f"AUC={roc_auc:.3f}"

)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.savefig(
    "roc_curve.png"
)

plt.show()


# =====================================================
# TRANSACTION NETWORK GRAPH

try:
    network_graph = load_transaction_network(ROOT_DIR)
    plot_transaction_network(network_graph)
except Exception as error:
    print(
        "Unable to render transaction network graph:",
        error
    )


# =====================================================
# VGAE LOSS
# =====================================================

plt.figure()

plt.plot(
    vgae_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "VGAE Loss Curve"
)

plt.savefig(
    "vgae_loss.png"
)

plt.show()


# =====================================================
# RGCN LOSS
# =====================================================

plt.figure()

plt.plot(
    rgcn_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "RGCN Loss Curve"
)

plt.savefig(
    "rgcn_loss.png"
)

plt.show()


# =====================================================
# FUSION LOSS
# =====================================================

plt.figure()

plt.plot(
    fusion_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Fusion Loss Curve"
)

plt.savefig(
    "fusion_loss.png"
)

plt.show()


print("\nAll images saved successfully")
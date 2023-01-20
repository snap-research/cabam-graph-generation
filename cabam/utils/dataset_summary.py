from collections import Counter

import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import networkx as nx
import pickle
import logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import powerlaw


def test_degree_distribution(degrees, xmin=None):
    if xmin:
        results = powerlaw.Fit(degrees, xmin=xmin)
    else:
        results = powerlaw.Fit(degrees)
    return results.power_law.alpha, results.power_law.xmin


def make_undirected_and_self_looped(A):
    """Takes scipy sparse matrix"""
    A_hat = A.maximum(A.T)  # make symmetric
    A_hat.setdiag(1)  # populate diagonal with self-loops
    return A_hat


def load_g2g_dataset(dataset: str):
    """Load a graph from a Numpy binary file.
    Adapted from https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py
    """

    path = f"data/{dataset}/raw/{dataset}.npz"
    if not path.endswith(".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        H = csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )
        y = loader.get("labels")
        A = make_undirected_and_self_looped(A)
        A = A.toarray()
        H = H.toarray()
        return A, H, y


def load_gaug_dataset(dataset: str):
    with open(
            f"data/alternative/{dataset}_features.pkl", "rb"
    ) as f_obj, open(
        f"data/alternative/{dataset}_adj.pkl", "rb"
    ) as a_obj, open(
        f"data/alternative/{dataset}_labels.pkl", "rb"
    ) as l_obj:
        features = pickle.load(f_obj)
        adj = pickle.load(a_obj)
        labels = pickle.load(l_obj)

    if sp.issparse(features):
        H = features.toarray()
    else:
        H = features.numpy()

    if sp.issparse(adj):
        A = adj.toarray()
    else:
        A = adj

    if type(labels) != np.ndarray:
        y = labels.numpy().ravel()
    else:
        y = labels.ravel()

    return A, H, y


def produce_processed_data(dataset: str):
    logging.info(f"Loading dataset {dataset}")
    if dataset in ["airport", "flickr", "blogcatalog"]:
        A, H, y = load_gaug_dataset(dataset=dataset)
    else:
        A, H, y = load_g2g_dataset(dataset=dataset)
    H[np.isnan(H)] = 0
    G = nx.from_numpy_matrix(A)
    return A, H, y, G


def plot_degree_label_assortativity(G, y, dataset=None, bins=10):
    degs = []
    assortativities = []
    for node in G.nodes():
        deg = G.degree(node)
        neighbor_labels = [int(y[i] == y[node]) for i in list(G.neighbors(node))]
        assortativity = np.sum(neighbor_labels) / len(neighbor_labels)
        degs.append(deg)
        assortativities.append(assortativity)
    degs = np.array(degs)
    assortativities = np.array(assortativities)

    # visualize
    boxplot_arrs = []
    boxplot_labels = []
    deg_bins = np.logspace(np.log10(min(degs)), np.log10(max(degs) + 1), bins)
    for deg_st, deg_end in zip(deg_bins, deg_bins[1:]):
        boxplot_arrs.append(assortativities[(degs >= deg_st) & (degs < deg_end)])
        boxplot_labels.append(f"{int(deg_end)}")

    plt.figure()
    plt.boxplot(boxplot_arrs, labels=boxplot_labels)
    if dataset:
        plt.title(dataset[0].upper() + dataset[1:], fontsize=16)
    plt.xlabel("Degree", fontsize=16)
    plt.ylabel("Class assortativity", fontsize=16)
    plt.savefig(
        f"figures/{dataset}_class-assortativity.png",
        bbox_inches="tight",
        dpi=150,
    )
    return degs, assortativities


def compute_overall_label_assortativity(G, y):
    intra_class = 0.0
    inter_class = 0.0
    for src, dst in G.edges():
        if y[src] == y[dst]:
            intra_class += 1
        else:
            inter_class += 1

    assortativity = intra_class / (intra_class + inter_class)
    logging.info(f"Overall assortativity: {assortativity}")
    return assortativity


def plot_node_level_assortativity(G, y, dataset=None):
    assortativities = []

    for node in G.nodes():
        same_class = 0
        cross_class = 0
        neighbors = np.array(list(G.neighbors(node)))

        for neighbor in neighbors:
            if y[node] == y[neighbor]:
                same_class += 1
            else:
                cross_class += 1
        assortativity = float(same_class) / (same_class + cross_class)
        assortativities.append(assortativity)

    plt.figure()
    plt.hist(assortativities, rwidth=0.8)
    plt.xlabel("Assortativity")
    plt.ylabel("Count")
    plt.title(dataset.upper())
    plt.savefig(f"figures/{dataset}.png")


def log_dataset_summary(A, H):
    nodes = A.shape[0]
    edges = A.sum()
    features = H.shape[1]
    logging.info(f"Nodes: {nodes}")
    logging.info(f"Edges: {edges}")
    logging.info(f"Features: {features}")
    return nodes, edges, features


def plot_degree_distribution(A, dataset):
    degs = A.sum(axis=1)
    c = Counter(degs)
    degs = sorted(c.items())
    plt.figure()
    plt.loglog([deg for (deg, ct) in degs], [ct for (deg, ct) in degs])
    plt.title(dataset[0].upper() + dataset[1:], fontsize=16)
    plt.xlabel("Degree", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.savefig(f"figures/{dataset}_degree.png", bbox_inches="tight", dpi=150)


def emb_degree_prediction(A, H):
    d = A.sum(axis=1)
    d = (d > np.mean(d)).astype(int)
    scores = []
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(H, d):
        clf = RandomForestClassifier().fit(H[train_idx, :], d[train_idx])
        dhat = clf.predict_proba(H[test_idx, :])[:, 1]
        scores.append(roc_auc_score(d[test_idx], dhat))

    auc_val = np.mean(scores)
    logging.info(f'Above mean degree prediction: {auc_val}')
    return auc_val


def draw_network_with_labels(G, node_labels):
    """
    Given a networkx graph and labels, draw nodes with relevant colors.

    G: networkx graph on N nodes
    node_labels: label vector of length N (assuming K labels, this would be labeled 0...K-1)

    Note: color_mapping currently fixed to 20 colors cyclically, but can trivially be extended.
    """

    unique_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
        '#ffffff', '#000000'
    ]
    color_mapping = {i: color for i, color in enumerate(unique_colors)}
    node_colors = [color_mapping[i % len(unique_colors)] for i in node_labels]
    nx.draw_kamada_kawai(G, with_labels=False, node_size=50, node_color=node_colors)

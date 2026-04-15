import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap

from sklearn.cluster import (KMeans, AffinityPropagation, DBSCAN, MeanShift)

warnings.filterwarnings('ignore')

SEED = 42
PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
           '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
           '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3']


def ase(X, labels, centers):
    """Average Squared Error: mean squared distance of each point to its center."""
    total = 0.0
    for c in range(len(centers)):
        members = X[labels == c]
        if len(members):
            total += np.sum((members - centers[c]) ** 2)
    return total / len(X)


def run_kmeans(X, n_clusters, n_init=10):
    m = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=SEED)
    m.fit(X)
    return m.labels_, m.cluster_centers_


def run_kcenters(X, n_clusters):
    centers_idx = [0]
    min_dists = np.linalg.norm(X - X[centers_idx[0]], axis=1)

    for _ in range(1, n_clusters):
        new_center_idx = np.argmax(min_dists)
        centers_idx.append(new_center_idx)
        new_dists = np.linalg.norm(X - X[new_center_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    centers = X[centers_idx]
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
    return labels, centers


def run_ap(X, max_iter=200):
    m = AffinityPropagation(random_state=SEED, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X)
    return m.labels_


def run_dbscan(X, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def run_meanshift(X):
    m = MeanShift()
    m.fit(X)
    return m.labels_

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsection(title):
    print(f"\n--- {title} ---")

def cluster_cmap(n):
    return ListedColormap(PALETTE[:n])
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from src.knets import KNets

def _geodesic_distance_matrix(X, n_neighbors=10):
    n = X.shape[0]
    euc = cdist(X, X, metric='euclidean')

    row, col, data = [], [], []
    for i in range(n):
        d_i = euc[i].copy()
        d_i[i] = np.inf
        nn_idx = np.argpartition(d_i, n_neighbors)[:n_neighbors]
        for j in nn_idx:
            row.append(i)
            col.append(j)
            data.append(euc[i, j])

    graph = csr_matrix((data, (row, col)), shape=(n, n))
    graph = graph.maximum(graph.T)   # make symmetric

    geo_dist = shortest_path(graph, method='D', directed=False)

    if np.any(np.isinf(geo_dist)):
        import warnings
        warnings.warn(
            f"Geodesic distance graph is disconnected for "
            f"n_neighbors={n_neighbors}. Increase n_neighbors or switch "
            "to the parallel architecture. Falling back to Euclidean.",
            RuntimeWarning
        )
        return euc

    return geo_dist

class SerialTwoLayerKNets:
    def __init__(self, k1=5, k2=5, n_clusters=None,
                 geo_neighbors=10, max_iter=10):
        self.k1 = k1
        self.k2 = k2
        self.n_clusters = n_clusters
        self.geo_neighbors = geo_neighbors
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.layer1_ = KNets(k=self.k1, max_iter=self.max_iter,
                             metric='euclidean')
        self.layer1_.fit(X)
        layer1_labels = self.layer1_.labels_ 
        E1 = self.layer1_.cluster_centers_

        geo_dist = _geodesic_distance_matrix(E1, self.geo_neighbors)
        self.layer2_ = KNets(k=self.k2, n_clusters=self.n_clusters,
                             max_iter=self.max_iter, metric='euclidean')

        if self.n_clusters is None:
            exemplar_idx = self.layer2_._nom(geo_dist, self.k2)
        else:
            exemplar_idx = self.layer2_._eom(geo_dist, self.k2,
                                             self.n_clusters)

        layer2_labels, layer2_centers, _ = \
            self.layer2_._assignment_phase(E1, geo_dist, exemplar_idx)

        self.layer2_.labels_ = layer2_labels
        self.layer2_.cluster_centers_ = layer2_centers
        self.layer2_.n_clusters_ = len(layer2_centers)

        self.labels_ = layer2_labels[layer1_labels]
        self.cluster_centers_ = layer2_centers
        self.n_clusters_ = self.layer2_.n_clusters_
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        if not hasattr(self, 'labels_'):
            raise ValueError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        dists = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(dists, axis=1)


class ParallelTwoLayerKNets:
    def __init__(self, k1=5, k2=5, n_subsets=None,
                 n_clusters=None, max_iter=10):
        self.k1 = k1
        self.k2 = k2
        self.n_subsets = n_subsets
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if self.n_subsets is None:
            self.n_subsets_ = max(2, n // 500)
        else:
            self.n_subsets_ = self.n_subsets

        subsets = np.array_split(np.arange(n), self.n_subsets_)
        pooled_exemplars = []

        for subset_idx in subsets:
            X_sub = X[subset_idx]
            layer1 = KNets(k=self.k1, max_iter=self.max_iter,
                           metric='euclidean')
            layer1.fit(X_sub)
            pooled_exemplars.append(layer1.cluster_centers_)

        E_pool = np.vstack(pooled_exemplars)

        self.layer2_ = KNets(k=self.k2, n_clusters=self.n_clusters,
                             max_iter=self.max_iter, metric='euclidean')
        self.layer2_.fit(E_pool)

        dists = cdist(X, self.layer2_.cluster_centers_, metric='euclidean')
        self.labels_ = np.argmin(dists, axis=1)
        self.cluster_centers_ = self.layer2_.cluster_centers_
        self.n_clusters_ = self.layer2_.n_clusters_
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        if not hasattr(self, 'labels_'):
            raise ValueError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        dists = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(dists, axis=1)
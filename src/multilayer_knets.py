import numpy as np
from scipy.spatial.distance import cdist
from src.knets import KNets


class SerialTwoLayerKNets:
    def __init__(self, k1=5, k2=5, n_clusters=None, geo_k=10, max_iter=10, use_geodesic=True):
        self.k1 = k1
        self.k2 = k2
        self.n_clusters = n_clusters
        self.geo_k = geo_k
        self.max_iter = max_iter
        self.use_geodesic = use_geodesic

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.layer1_ = KNets(k=self.k1,max_iter=self.max_iter,metric="euclidean").fit(X)

        E1 = self.layer1_.cluster_centers_
        layer1_labels = self.layer1_.labels_

        self.layer2_ = KNets(
            k=self.k2,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            metric="geodesic" if self.use_geodesic else "euclidean",
            geo_k=self.geo_k
        )

        self.layer2_.fit(E1)

        self.labels_ = self.layer2_.labels_[layer1_labels]
        self.cluster_centers_ = self.layer2_.cluster_centers_
        self.n_clusters_ = self.layer2_.n_clusters_

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = cdist(X, self.cluster_centers_, metric="euclidean")
        return np.argmin(dists, axis=1)


class ParallelTwoLayerKNets:
    def __init__(self, k1=5, k2=5, n_subsets=None, n_clusters=None, max_iter=10):
        self.k1 = k1
        self.k2 = k2
        self.n_subsets = n_subsets
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        n_subsets = self.n_subsets or max(2, n // 500)
        subsets = np.array_split(np.arange(n), n_subsets)

        pooled = []

        for idx in subsets:
            X_sub = X[idx]

            model = KNets(
                k=self.k1,
                max_iter=self.max_iter,
                metric="euclidean"
            ).fit(X_sub)

            pooled.append(model.cluster_centers_)

        E_pool = np.vstack(pooled)

        self.layer2_ = KNets(
            k=self.k2,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            metric="euclidean"
        ).fit(E_pool)

        self.cluster_centers_ = self.layer2_.cluster_centers_

        dists = cdist(X, self.cluster_centers_, metric="euclidean")
        self.labels_ = np.argmin(dists, axis=1)
        self.n_clusters_ = self.layer2_.n_clusters_

        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = cdist(X, self.cluster_centers_, metric="euclidean")
        return np.argmin(dists, axis=1)

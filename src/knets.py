import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

class KNets:
    def __init__(self, k=5, n_clusters=None, max_iter=10, metric='euclidean', geo_k=None):
        self.k = k
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.geo_k = geo_k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        print(f"[KNets] fit called | n={n}, k={self.k}, "
              f"mode={'EOM' if self.n_clusters else 'NOM'}"
              + (f", n_clusters={self.n_clusters}" if self.n_clusters else ""))

        dist = self._compute_distance_matrix(X)
        self.dist_ = dist

        if self.n_clusters is None:
            exemplar_idx = self._nom(dist, self.k)
        else:
            exemplar_idx = self._eom(dist, self.k, self.n_clusters)

        self.labels_, self.cluster_centers_, self.n_iter_ = self._assignment_phase(
            X, dist, exemplar_idx
        )

        self.n_clusters_ = len(self.cluster_centers_)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def _compute_distance_matrix(self, X):
        X = np.asarray(X, dtype=float)

        if self.metric == 'euclidean':
            return cdist(X, X, metric='euclidean')

        elif self.metric == 'geodesic':
            graph = kneighbors_graph(
                X,
                n_neighbors=self.geo_k,
                mode='distance',
                include_self=False
            )

            dist = shortest_path(graph, directed=False)

            if np.isinf(dist).any():
                return cdist(X, X, metric='euclidean')

            return dist

    def _build_pre_clusters(self, dist, k_val):
        n = dist.shape[0]

        sorted_neighbors = np.argsort(dist, axis=1)

        has_self_at_zero = sorted_neighbors[:, 0] == np.arange(n)
        if has_self_at_zero.all():
            sorted_neighbors = sorted_neighbors[:, 1:]
        else:
            mask = sorted_neighbors != np.arange(n)[:, None]
            sorted_neighbors = np.array([
                sorted_neighbors[i][mask[i]] for i in range(n)
            ])

        knn = sorted_neighbors[:, :k_val]
        kth_dists = dist[np.arange(n), knn[:, -1]]

        knn_dists = dist[np.arange(n)[:, None], knn]
        scores_raw = knn_dists.sum(axis=1)

        pre_clusters = []
        scores = np.empty(n)

        for i in range(n):
            beyond = sorted_neighbors[i, k_val:]
            ties = beyond[dist[i, beyond] == kth_dists[i]]
            members = np.concatenate([[i], knn[i], ties])
            pre_clusters.append(set(members.tolist()))
            scores[i] = scores_raw[i] / len(members)

        return pre_clusters, scores

    def _selection_phase(self, pre_clusters, scores, dist, locked_exemplars=None):
        if locked_exemplars is None:
            locked_exemplars = set()

        scores = self._resolve_instabilities(pre_clusters, scores, dist)

        sorted_idx = np.argsort(scores)
        covered = set()
        exemplar_indices = list(locked_exemplars)

        for idx in locked_exemplars:
            covered.update(pre_clusters[idx])

        for idx in sorted_idx:
            if idx in locked_exemplars:
                continue
            if pre_clusters[idx].isdisjoint(covered):
                exemplar_indices.append(idx)
                covered.update(pre_clusters[idx])

        return exemplar_indices

    def _resolve_instabilities(self, pre_clusters, scores, dist):
        scores = scores.copy()

        _, inverse, counts = np.unique(
            scores, return_inverse=True, return_counts=True
        )

        for uid in np.where(counts >= 2)[0]:
            tied_indices = np.where(inverse == uid)[0]

            members_array = [pre_clusters[i] for i in tied_indices]
            has_overlap = any(
                not members_array[a].isdisjoint(members_array[b])
                for a in range(len(tied_indices))
                for b in range(a + 1, len(tied_indices))
            )

            if not has_overlap:
                continue

            global_scores = dist[np.ix_(tied_indices, np.arange(dist.shape[0]))]
            mask = np.ones(dist.shape[0], dtype=bool)

            for local, idx in enumerate(tied_indices):
                mask[idx] = False
                scores[idx] = global_scores[local, mask].mean()
                mask[idx] = True

        return scores

    def _nom(self, dist, k_val):
        pre_clusters, scores = self._build_pre_clusters(dist, k_val)
        return self._selection_phase(pre_clusters, scores, dist)

    def _eom(self, dist, k_start, n_clusters_requested):
        n = dist.shape[0]
        k_val = k_start
        locked = set()
        locked_scores = {}

        while len(locked) < n_clusters_requested and k_val >= 1:
            pre_clusters, scores = self._build_pre_clusters(dist, k_val)
            sub_scores = self._resolve_instabilities(pre_clusters, scores, dist)

            new_exemplars = self._selection_phase(
                pre_clusters, sub_scores, dist, locked_exemplars=locked
            )

            for idx in new_exemplars:
                if idx not in locked:
                    locked.add(idx)
                    locked_scores[idx] = sub_scores[idx]

            k_val -= 1

        ranked = sorted(locked, key=lambda i: locked_scores.get(i, np.inf))
        return ranked[:n_clusters_requested]

    def _assign_to_centers(self, centers_idx):
        return np.argmin(self.dist_[:, centers_idx], axis=1)

    def _assignment_phase(self, X, dist, exemplar_idx):
        center_indices = list(exemplar_idx)
        labels = self._assign_to_centers(center_indices)
        n_iter = 0

        for iteration in range(self.max_iter):
            new_center_indices = []

            for c in range(len(center_indices)):
                member_mask = labels == c

                if not np.any(member_mask):
                    new_center_indices.append(center_indices[c])
                    continue

                member_idx = np.where(member_mask)[0]
                sub_dist = dist[np.ix_(member_idx, member_idx)]
                local_best = np.argmin(sub_dist.sum(axis=1))
                new_center_indices.append(member_idx[local_best])

            if new_center_indices == center_indices:
                break

            center_indices = new_center_indices
            labels = self._assign_to_centers(center_indices)
            n_iter = iteration + 1

        centers = X[center_indices]
        return labels, centers, n_iter

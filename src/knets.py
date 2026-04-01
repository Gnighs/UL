import numpy as np
from scipy.spatial.distance import cdist


class KNets:
    def __init__(self, k=5, n_clusters=None, max_iter=10, metric='euclidean'):
        self.k = k
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        print(f"[KNets] fit called | n={n}, k={self.k}, "
              f"mode={'EOM' if self.n_clusters else 'NOM'}"
              + (f", n_clusters={self.n_clusters}" if self.n_clusters else ""))

        print("[KNets] computing distance matrix...")
        dist = self._compute_distance_matrix(X)

        if self.n_clusters is None:
            exemplar_idx = self._nom(dist, self.k)
        else:
            exemplar_idx = self._eom(dist, self.k, self.n_clusters)

        print(f"[KNets] assignment phase | {len(exemplar_idx)} pre-exemplars...")
        self.labels_, self.cluster_centers_, self.n_iter_ = \
            self._assignment_phase(X, dist, exemplar_idx)
        self.n_clusters_ = len(self.cluster_centers_)
        print(f"[KNets] done | clusters={self.n_clusters_}, iters={self.n_iter_}")
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self._assign_to_centers(X, self.cluster_centers_)

    def _compute_distance_matrix(self, X):
        return cdist(X, X, metric=self.metric)

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
            K = len(members)
            pre_clusters.append(set(members.tolist()))
            scores[i] = scores_raw[i] / K

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
        n = len(scores)

        unique_scores, inverse, counts = np.unique(
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

            global_scores = dist[np.ix_(tied_indices, np.arange(n))]
            mask = np.ones(n, dtype=bool)
            for local, idx in enumerate(tied_indices):
                mask[idx] = False
                scores[idx] = global_scores[local, mask].mean()
                mask[idx] = True

        return scores

    def _nom(self, dist, k_val):
        print(f"[KNets] NOM | building pre-clusters k={k_val}...")
        pre_clusters, scores = self._build_pre_clusters(dist, k_val)
        print(f"[KNets] NOM | selection phase...")
        exemplars = self._selection_phase(pre_clusters, scores, dist)
        print(f"[KNets] NOM | {len(exemplars)} pre-exemplars selected")
        return exemplars

    def _eom(self, dist, k_start, n_clusters_requested):
        n = dist.shape[0]
        k_val = k_start
        locked = set()
        locked_scores = {}
        all_pre_clusters = [None] * n
        epoch = 0

        while len(locked) < n_clusters_requested and k_val >= 1:
            epoch += 1
            print(f"[KNets] EOM | epoch={epoch}, k={k_val}, "
                  f"exemplars so far={len(locked)}/{n_clusters_requested}")

            pre_clusters_epoch, scores_epoch = self._build_pre_clusters(dist, k_val)
            sub_scores = self._resolve_instabilities(pre_clusters_epoch, scores_epoch, dist)

            new_exemplars = self._selection_phase(
                pre_clusters_epoch, sub_scores, dist, locked_exemplars=locked
            )

            newly_added = [idx for idx in new_exemplars if idx not in locked]
            for idx in newly_added:
                locked.add(idx)
                locked_scores[idx] = sub_scores[idx]
                all_pre_clusters[idx] = pre_clusters_epoch[idx]

            k_val -= 1

        print(f"[KNets] EOM | finished in {epoch} epochs, "
              f"{len(locked)} exemplars collected")
        ranked = sorted(locked, key=lambda i: locked_scores.get(i, np.inf))
        return ranked[:n_clusters_requested]

    def _assign_to_centers(self, X, centers):
        dists = cdist(X, centers, metric=self.metric)
        return np.argmin(dists, axis=1)

    def _assignment_phase(self, X, dist, exemplar_idx):
        center_indices = list(exemplar_idx)
        centers = X[center_indices]
        labels = self._assign_to_centers(X, centers)
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
            centers = X[center_indices]
            labels = self._assign_to_centers(X, centers)
            n_iter = iteration + 1

        return labels, centers, n_iter
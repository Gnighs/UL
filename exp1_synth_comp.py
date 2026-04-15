import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import (SEED, run_kmeans, run_ap, run_dbscan, run_meanshift, subsection)
from src.knets import KNets


datasets = []

X_blobs, y_blobs = make_blobs(n_samples=600, centers=31, cluster_std=0.4,
                                random_state=SEED)
datasets.append(("Isotropic blobs\n(31 clusters)", X_blobs, y_blobs, 31))

X_aniso, y_aniso = make_blobs(n_samples=400, centers=4, random_state=SEED)
T = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = X_aniso @ T
datasets.append(("Non-isotropic blobs\n(4 clusters)", X_aniso, y_aniso, 4))

X_var, y_var = make_blobs(n_samples=[100, 200, 400],
                            centers=[[0, 0], [3, 2], [6, 0]],
                            random_state=SEED)
datasets.append(("Variable-density blobs\n(3 clusters)", X_var, y_var, 3))

X_moons, y_moons = make_moons(n_samples=400, noise=0.07, random_state=SEED)
datasets.append(("Moons\n(2 clusters)", X_moons, y_moons, 2))

X_circ, y_circ = make_circles(n_samples=400, noise=0.05, factor=0.5,
                                random_state=SEED)
datasets.append(("Circles\n(2 clusters)", X_circ, y_circ, 2))

k_vals   = [8,   5,   5,  5,  5]
eps_vals = [0.4, 0.6, 1.5, 0.3, 0.25]

algo_names = ["K-Nets EOM", "K-Means", "AP", "DBSCAN", "MeanShift"]
fig, axes = plt.subplots(len(datasets), len(algo_names),
                            figsize=(18, len(datasets) * 3.2))
fig.suptitle("EXP 1 — Synthetic datasets: clustering comparison\n"
                "(approximates Figure 5 of the paper)", fontsize=13, y=1.01)

for di, (name, X, y_true, nc) in enumerate(datasets):
    Xs = StandardScaler().fit_transform(X)
    subsection(name.replace('\n', ' '))
    k   = k_vals[di]
    eps = eps_vals[di]

    t0 = time.time()
    km = KNets(k=k, n_clusters=nc)
    labels_kn = km.fit_predict(Xs)
    print(f"  KNets EOM   | C={km.n_clusters_:3d} | NMI={NMI(y_true, labels_kn):.3f} | t={time.time()-t0:.2f}s")

    t0 = time.time()
    labels_km, _ = run_kmeans(Xs, nc)
    print(f"  K-Means     | C={nc:3d} | NMI={NMI(y_true, labels_km):.3f} | t={time.time()-t0:.2f}s")

    t0 = time.time()
    labels_ap = run_ap(Xs)
    nc_ap = len(np.unique(labels_ap))
    print(f"  AP          | C={nc_ap:3d} | NMI={NMI(y_true, labels_ap):.3f} | t={time.time()-t0:.2f}s")

    t0 = time.time()
    labels_db = run_dbscan(Xs, eps=eps)
    nc_db = len(np.unique(labels_db[labels_db >= 0]))
    print(f"  DBSCAN      | C={nc_db:3d} | NMI={NMI(y_true, labels_db):.3f} | t={time.time()-t0:.2f}s")

    t0 = time.time()
    labels_ms = run_meanshift(Xs)
    nc_ms = len(np.unique(labels_ms))
    print(f"  MeanShift   | C={nc_ms:3d} | NMI={NMI(y_true, labels_ms):.3f} | t={time.time()-t0:.2f}s")

    all_labels = [labels_kn, labels_km, labels_ap, labels_db, labels_ms]

    for ai, (alg_name, lbls) in enumerate(zip(algo_names, all_labels)):
        ax = axes[di][ai]
        nc_plot = len(np.unique(lbls[lbls >= 0]))
        cmap = plt.cm.get_cmap('tab20', max(nc_plot, 2))
        ax.scatter(Xs[:, 0], Xs[:, 1], c=lbls, cmap=cmap, s=8, linewidths=0)
        if di == 0:
            ax.set_title(alg_name, fontsize=10, fontweight='bold')
        if ai == 0:
            ax.set_ylabel(name, fontsize=8)
        nmi_v = NMI(y_true, lbls)
        ax.set_xlabel(f"NMI={nmi_v:.2f} C={len(np.unique(lbls[lbls >= 0]))}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"outputs/exp1_synthetic_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] exp1_synthetic_comparison.png")
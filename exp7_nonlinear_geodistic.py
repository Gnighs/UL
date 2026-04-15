import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import SEED, run_kmeans, run_ap, run_dbscan, section, subsection
from src.knets import KNets
from src.multilayer_knets import SerialTwoLayerKNets

nonlinear_sets = [
    ("Moons",   *make_moons(n_samples=500,   noise=0.06,              random_state=SEED), 2),
    ("Circles", *make_circles(n_samples=500, noise=0.04, factor=0.45, random_state=SEED), 2),
]

fig, axes = plt.subplots(len(nonlinear_sets), 5,
                            figsize=(18, len(nonlinear_sets) * 3.5))
fig.suptitle("EXP 7 — Non-linear datasets: single-layer vs two-layer serial K-Nets + geodesic\n"
                "(approximates Fig. 5 V and VI)", fontsize=11, y=1.02)

for di, (name, X, y_true, nc) in enumerate(nonlinear_sets):
    Xs = StandardScaler().fit_transform(X)
    subsection(name)

    # Single-layer K-Nets EOM (Euclidean)
    m_single = KNets(k=5, n_clusters=nc)
    lbls_single = m_single.fit_predict(Xs)
    nmi_single = NMI(y_true, lbls_single)
    print(f"  Single KNets EOM   NMI={nmi_single:.3f}")

    # Two-layer serial with geodesic — try multiple geo_neighbors, keep best
    nmi_serial_best = 0
    lbls_serial_best = None
    for gn in [5, 8, 12]:
        try:
            m_ser = SerialTwoLayerKNets(k1=3, k2=3, n_clusters=nc, geo_neighbors=gn)
            lbls_ser = m_ser.fit_predict(Xs)
            nmi_ser = NMI(y_true, lbls_ser)
            print(f"  Serial geo_n={gn:2d}      NMI={nmi_ser:.3f} | L1 exemplars={m_ser.layer1_.n_clusters_}")
            if nmi_ser > nmi_serial_best:
                nmi_serial_best = nmi_ser
                lbls_serial_best = lbls_ser
        except Exception as e:
            print(f"  Serial geo_n={gn} failed: {e}")

    # K-Means
    lbls_km, _ = run_kmeans(Xs, nc)
    nmi_km = NMI(y_true, lbls_km)
    print(f"  K-Means            NMI={nmi_km:.3f}")

    # AP
    lbls_ap = run_ap(Xs)
    nmi_ap = NMI(y_true, lbls_ap)
    print(f"  AP                 NMI={nmi_ap:.3f} C={len(np.unique(lbls_ap))}")

    # DBSCAN (tuned per dataset)
    eps_nl = 0.25 if name == "Moons" else 0.2
    lbls_db = run_dbscan(Xs, eps=eps_nl, min_samples=5)
    nmi_db = NMI(y_true, lbls_db)
    print(f"  DBSCAN             NMI={nmi_db:.3f} C={len(np.unique(lbls_db[lbls_db >= 0]))}")

    plot_data = [
        ("K-Nets\nSingle-layer",     lbls_single,
            nmi_single),
        ("K-Nets\nSerial+Geodesic",  lbls_serial_best if lbls_serial_best is not None else lbls_single,
            nmi_serial_best),
        ("K-Means",                  lbls_km,  nmi_km),
        ("AP",                       lbls_ap,  nmi_ap),
        ("DBSCAN",                   lbls_db,  nmi_db),
    ]

    for ai, (alg_name, lbls, nmi_v) in enumerate(plot_data):
        ax = axes[di][ai] if len(nonlinear_sets) > 1 else axes[ai]
        cmap = plt.cm.get_cmap('tab10', max(len(np.unique(lbls[lbls >= 0])), 2))
        ax.scatter(Xs[:, 0], Xs[:, 1], c=lbls, cmap=cmap, s=8, linewidths=0)
        if di == 0:
            ax.set_title(alg_name, fontsize=9, fontweight='bold')
        if ai == 0:
            ax.set_ylabel(name, fontsize=9)
        ax.set_xlabel(f"NMI={nmi_v:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"outputs/exp7_nonlinear_geodesic.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] exp7_nonlinear_geodesic.png")
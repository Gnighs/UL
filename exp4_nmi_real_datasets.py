import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import run_kmeans, run_ap, subsection
from src.knets import KNets


d = load_digits()
dw = load_wine()
di = load_iris()
db = load_breast_cancer()

mask = np.isin(d.target, [5, 6, 7, 8, 9])
X_d59, y_d59 = d.data[mask], d.target[mask]

real_datasets = [
    ("Digits 5-9\n(proxy MNIST 5-9)", X_d59, y_d59, 5),
    ("Digits 0-9\n(proxy pen-digits)", d.data, d.target, 10),
    ("Wine\n(3 classes)", dw.data, dw.target, 3),
    ("Iris\n(3 classes)", di.data, di.target, 3),
    ("Breast Cancer\n(2 classes)", db.data, db.target, 2),
]

results_table = []

for ds_name, X, y_true, nc in real_datasets:
    Xs = StandardScaler().fit_transform(X)
    subsection(ds_name.replace('\n', ' '))
    row = {"Dataset": ds_name.replace('\n', ' ')}

    # 1. KNets Geodesic
    t0 = time.time()
    m_geo = KNets(
        k=max(5, nc * 2),
        n_clusters=nc,
        metric='geodesic',
        geo_k=10
    )
    m_geo.fit(Xs)
    row["KNets EOM (Geodesic)"] = NMI(y_true, m_geo.labels_)
    row["KNets_geo_t"] = time.time() - t0
    print(f"  KNets GEO   NMI={row['KNets EOM (Geodesic)']:.3f}  t={row['KNets_geo_t']:.2f}s")

    # 2. KNets Euclidean
    t0 = time.time()
    m = KNets(
        k=max(5, nc * 2),
        n_clusters=nc,
        metric='euclidean'
    )
    m.fit(Xs)
    row["KNets EOM (Euclidean)"] = NMI(y_true, m.labels_)
    row["KNets_euc_t"] = time.time() - t0
    print(f"  KNets EUC   NMI={row['KNets EOM (Euclidean)']:.3f}  t={row['KNets_euc_t']:.2f}s")

    # 3. AP
    t0 = time.time()
    lbls_ap = run_ap(Xs)
    row["AP"] = NMI(y_true, lbls_ap)
    row["AP_t"] = time.time() - t0
    print(f"  AP          NMI={row['AP']:.3f}  t={row['AP_t']:.2f}s")

    # 4. K-Means (LAST)
    t0 = time.time()
    lbls_km, _ = run_kmeans(Xs, nc, n_init=10)
    row["K-Means"] = NMI(y_true, lbls_km)
    row["KMeans_t"] = time.time() - t0
    print(f"  K-Means     NMI={row['K-Means']:.3f}  t={row['KMeans_t']:.2f}s")

    results_table.append(row)

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ds_names = [r["Dataset"] for r in results_table]

algos = [
    "KNets EOM (Geodesic)",
    "KNets EOM (Euclidean)",
    "AP",
    "K-Means"
]

colors = ['#2E5299', '#1F3864', '#f58231', '#e6194b']

x = np.arange(len(ds_names))
width = 0.2

for i, (algo, color) in enumerate(zip(algos, colors)):
    vals = [r.get(algo, 0) for r in results_table]
    bars = ax.bar(x + i * width - width * 1.5, vals, width,
                  label=algo, color=color, alpha=0.85)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2f}",
            ha='center',
            va='bottom',
            fontsize=6.5,
            rotation=90
        )

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=9)
ax.set_ylabel("NMI")
ax.set_ylim(0, 1.15)
ax.set_title("EXP 4 — NMI comparison on real datasets\nHigher is better")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("outputs/exp4_nmi_real_datasets.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n[saved] outputs/exp4_nmi_real_datasets.png")

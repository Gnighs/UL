import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import ase, run_kcenters
from src.knets import KNets

d = load_digits()
X_ase = StandardScaler().fit_transform(d.data)
y_ase = d.target

cluster_range = list(range(5, 51, 5))
ase_knets, ase_kcenters = [], []
nmi_knets, nmi_kcenters = [], []
time_knets, time_kcenters = [], []

print("Computing ASE and NMI: K-Nets vs K-Centers (single init)...")
for nc in cluster_range:
    print(f"  nc={nc:2d}...")

    # K-Nets EOM
    t0 = time.time()
    m = KNets(k=max(5, nc), n_clusters=nc)
    m.fit(X_ase)
    time_knets.append(time.time() - t0)
    ase_knets.append(ase(X_ase, m.labels_, m.cluster_centers_))
    nmi_knets.append(NMI(y_ase, m.labels_))

    # K-Centers (Gonzalez algorithm)
    t0 = time.time()
    lbls_kc, ctrs_kc = run_kcenters(X_ase, nc)
    time_kcenters.append(time.time() - t0)
    ase_kcenters.append(ase(X_ase, lbls_kc, ctrs_kc))
    nmi_kcenters.append(NMI(y_ase, lbls_kc))

ase_knets    = np.array(ase_knets)
ase_kcenters = np.array(ase_kcenters)
pct_improvement = 100 * (ase_kcenters - ase_knets) / ase_kcenters

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(cluster_range, ase_knets,    'o-', color='#1F3864', label='K-Nets EOM')
axes[0].plot(cluster_range, ase_kcenters, 's--', color='#e6194b', label='K-Centers (1 init)')
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("ASE (Lower is Better)")
axes[0].set_title("ASE vs cluster count")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].bar(cluster_range, pct_improvement, color='#2E5299', alpha=0.8, width=3.5)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xlabel("Number of clusters")
axes[1].set_ylabel("% ASE improvement (KNets vs KCenters)")
axes[1].set_title("K-Nets ASE improvement\n(positive = K-Nets better)")
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].plot(cluster_range, nmi_knets,    'o-', color='#1F3864', label='K-Nets EOM')
axes[2].plot(cluster_range, nmi_kcenters, 's--', color='#e6194b', label='K-Centers')
axes[2].set_xlabel("Number of clusters")
axes[2].set_ylabel("NMI (Higher is Better)")
axes[2].set_title("NMI vs cluster count (digits)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle("EXP 5 — ASE and NMI: K-Nets vs K-Centers (Gonzalez Algorithm)", fontsize=11)
plt.tight_layout()
plt.savefig(f"outputs/exp5_ase_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Mean ASE improvement of K-Nets over K-Centers: {pct_improvement.mean():.1f}%")
print(f"  Mean Time: KNets={np.mean(time_knets):.3f}s | KCenters={np.mean(time_kcenters):.3f}s")
print("[saved] exp5_ase_comparison.png")
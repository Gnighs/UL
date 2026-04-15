import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from src.helpers import ase
from src.knets import KNets

d = load_digits()
X_it = StandardScaler().fit_transform(d.data)

cluster_range_it = list(range(5, 41, 5))
ase_1iter, ase_full, iters_used = [], [], []

print("Comparing 1-iteration vs full convergence assignment phase...")
for nc in cluster_range_it:
    print(f"  nc={nc}...")

    m_full = KNets(k=max(5, nc), n_clusters=nc, max_iter=20)
    m_full.fit(X_it)
    ase_full.append(ase(X_it, m_full.labels_, m_full.cluster_centers_))
    iters_used.append(m_full.n_iter_)

    m_1 = KNets(k=max(5, nc), n_clusters=nc, max_iter=1)
    m_1.fit(X_it)
    ase_1iter.append(ase(X_it, m_1.labels_, m_1.cluster_centers_))

ase_1iter = np.array(ase_1iter)
ase_full  = np.array(ase_full)
pct_diff  = 100 * (ase_1iter - ase_full) / ase_full

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(cluster_range_it, ase_full,  'o-',  color='#1F3864', label='Full convergence')
ax1.plot(cluster_range_it, ase_1iter, 's--', color='#e6194b', label='1 iteration')
ax1.set_xlabel("Number of clusters"); ax1.set_ylabel("ASE")
ax1.set_title("ASE: 1 iteration vs full convergence")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(cluster_range_it, pct_diff, color='#f58231', alpha=0.85, width=3.5)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhline(2, color='red', linestyle='--', linewidth=1,
            label='Paper claim: ≤2% degradation')
ax2.set_xlabel("Number of clusters")
ax2.set_ylabel("% ASE degradation (1-iter vs full)")
ax2.set_title("% ASE degradation with 1 iteration\n(paper claims ≤2%)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')

ax3.bar(cluster_range_it, iters_used, color='#3cb44b', alpha=0.85, width=3.5)
ax3.axhline(5, color='red', linestyle='--', linewidth=1,
            label='Paper max observed: 5')
ax3.set_xlabel("Number of clusters")
ax3.set_ylabel("Iterations to convergence")
ax3.set_title("Actual iterations to convergence")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')

fig.suptitle("EXP 6 — Assignment phase iterations analysis (Fig. 7 style)", fontsize=11)
plt.tight_layout()
plt.savefig(f"outputs/exp6_iterations_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Max % ASE degradation with 1 iter: {pct_diff.max():.2f}%")
print(f"  Mean iterations to convergence: {np.mean(iters_used):.1f}")
print("[saved] exp6_iterations_analysis.png")
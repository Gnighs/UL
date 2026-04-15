import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import SEED
from src.knets import KNets
from src.multilayer_knets import ParallelTwoLayerKNets

sizes = [300, 600, 1000, 1500, 2000]
t_single, t_parallel = [], []
nmi_single_p, nmi_parallel_p = [], []

print("Comparing single-layer vs parallel two-layer K-Nets across dataset sizes...")
for n in sizes:
    X_p, y_p = make_blobs(n_samples=n, centers=10, cluster_std=0.8,
                            random_state=SEED)
    X_p = StandardScaler().fit_transform(X_p)
    print(f"  n={n}...")

    t0 = time.time()
    m_s = KNets(k=10, n_clusters=10)
    m_s.fit(X_p)
    t_single.append(time.time() - t0)
    nmi_single_p.append(NMI(y_p, m_s.labels_))

    t0 = time.time()
    n_sub = max(2, n // 200)
    m_p = ParallelTwoLayerKNets(k1=8, k2=8, n_clusters=10, n_subsets=n_sub)
    m_p.fit(X_p)
    t_parallel.append(time.time() - t0)
    nmi_parallel_p.append(NMI(y_p, m_p.labels_))

    print(f"    Single   t={t_single[-1]:.3f}s  NMI={nmi_single_p[-1]:.3f}")
    print(f"    Parallel t={t_parallel[-1]:.3f}s  NMI={nmi_parallel_p[-1]:.3f}")

speedup = [ts / tp for ts, tp in zip(t_single, t_parallel)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

ax1.plot(sizes, t_single,   'o-', color='#e6194b', label='Single-layer')
ax1.plot(sizes, t_parallel, 's-', color='#1F3864', label='Parallel two-layer')
ax1.set_xlabel("Dataset size (N)"); ax1.set_ylabel("Time (s)")
ax1.set_title("Runtime: single vs parallel two-layer")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(sizes, speedup, color='#3cb44b', alpha=0.85, width=sizes[0] * 0.6)
ax2.axhline(1, color='black', linewidth=0.8)
ax2.set_xlabel("Dataset size (N)")
ax2.set_ylabel("Speedup factor (single/parallel)")
ax2.set_title("Speedup factor of parallel two-layer\nvs single-layer")
ax2.grid(True, alpha=0.3, axis='y')

ax3.plot(sizes, nmi_single_p,   'o-', color='#e6194b', label='Single-layer')
ax3.plot(sizes, nmi_parallel_p, 's-', color='#1F3864', label='Parallel two-layer')
ax3.set_xlabel("Dataset size (N)"); ax3.set_ylabel("NMI")
ax3.set_title("NMI: single vs parallel two-layer\n(quality preservation check)")
ax3.legend(); ax3.grid(True, alpha=0.3)

fig.suptitle("EXP 8 — Parallel two-layer K-Nets: speedup and quality (Section 3.1)",
                fontsize=11)
plt.tight_layout()
plt.savefig(f"outputs/exp8_parallel_speedup.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Max speedup: {max(speedup):.2f}x at n={sizes[np.argmax(speedup)]}")
print("[saved] exp8_parallel_speedup.png")

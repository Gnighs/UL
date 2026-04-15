import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from src.helpers import SEED
from src.knets import KNets


X_sweep, _ = make_blobs(n_samples=600, centers=31, cluster_std=0.4,
                            random_state=SEED)
X_sweep = StandardScaler().fit_transform(X_sweep)

k_range = list(range(1, 51))
n_clusters_nom = []

print("Sweeping k for NOM on 31-cluster dataset...")
for k in k_range:
    m = KNets(k=k)
    m.fit(X_sweep)
    n_clusters_nom.append(m.n_clusters_)
    if k % 10 == 0:
        print(f"  k={k:3d} -> C={m.n_clusters_}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_range, n_clusters_nom, 'o-', color='#1F3864', markersize=4, linewidth=1.5)
ax.axhline(31, color='red', linestyle='--', linewidth=1, label='True clusters (31)')
ax.set_xlabel("Resolution parameter k")
ax.set_ylabel("Number of clusters (NOM)")
ax.set_title("EXP 2 — NOM landmark sweep\n"
                "Number of clusters found vs k on 31-cluster dataset")
ax.legend()
ax.grid(True, alpha=0.3)

true_ks = [k for k, c in zip(k_range, n_clusters_nom) if c == 31]
if true_ks:
    ax.axvspan(min(true_ks), max(true_ks), alpha=0.15, color='green',
                label=f'k range recovering 31 clusters: {min(true_ks)}–{max(true_ks)}')
    ax.legend()
    print(f"  True cluster count (31) recovered for k in [{min(true_ks)}, {max(true_ks)}]")

n_possible = len(X_sweep)
unique_counts = len(set(n_clusters_nom))
pct = 100 * unique_counts / n_possible
print(f"  Unique landmark partitions: {unique_counts} / {n_possible} = {pct:.1f}%")
ax.text(0.97, 0.95,
        f"Unique landmark partitions:\n{unique_counts}/{n_possible} = {pct:.1f}%",
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"outputs/exp2_nom_landmark_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print("[saved] exp2_nom_landmark_sweep.png")

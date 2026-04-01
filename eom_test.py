import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from knets import KNets
import sys

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

k_start = 30
target_clusters = int(sys.argv[1])

model = KNets(k=k_start, n_clusters=target_clusters).fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=model.labels_,
    cmap='plasma',
    s=40,
    edgecolors='k',
    linewidths=0.5
)

plt.title(
    f"K-Nets EOM (Exemplar Optimization Mode)\n"
    f"k_start={k_start} → target={target_clusters} | Found {model.n_clusters_} clusters"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

print(f"EOM requested {target_clusters} clusters → obtained {model.n_clusters_}")

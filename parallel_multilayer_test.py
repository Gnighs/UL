import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.multilayer_knets import ParallelTwoLayerKNets

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

model = ParallelTwoLayerKNets(
    k1=5,             # local clustering per subset
    k2=10,            # clustering on pooled exemplars
    n_subsets=4,      # split dataset
    n_clusters=2      # final clusters
)

labels = model.fit_predict(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap='plasma',
    s=30,
    edgecolors='k',
    linewidths=0.5
)

plt.title(
    f"Parallel Two-Layer KNets\n"
    f"subsets={model.n_subsets_}, k1={model.k1}, k2={model.k2}\n"
    f"Final clusters: {model.n_clusters_}"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

print(f"[Parallel] Found {model.n_clusters_} clusters")

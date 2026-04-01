import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.multilayer_knets import SerialTwoLayerKNets

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

model = SerialTwoLayerKNets(
    k1=5,              # local neighborhoods (layer 1)
    k2=10,             # neighborhood on exemplars
    n_clusters=2,      # final desired clusters (EOM behavior)
    geo_neighbors=10   # for geodesic distances
)

labels = model.fit_predict(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap='coolwarm',
    s=30,
    edgecolors='k',
    linewidths=0.5
)

plt.title(
    f"Serial Two-Layer KNets\n"
    f"k1={model.k1}, k2={model.k2}, geo_k={model.geo_neighbors}\n"
    f"Final clusters: {model.n_clusters_}"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

print(f"[Serial] Found {model.n_clusters_} clusters")

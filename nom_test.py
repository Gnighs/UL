import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from knets import KNets

print('Creating synthetic dataset with make_moons...')
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

k_val = 12
model = KNets(k=k_val).fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', s=40, edgecolors='k', linewidths=0.5)
plt.title(f"K-Nets NOM (Normal Operational Mode)\nFixed k={k_val} | Found {model.n_clusters_} clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
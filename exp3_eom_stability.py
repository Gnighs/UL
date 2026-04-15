import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as NMI

from src.helpers import SEED
from src.knets import KNets

print("Testing EOM recovery of 4 clusters from non-isotropic blobs...")
X_eom, y_eom = make_blobs(n_samples=400, centers=4, random_state=SEED)
T = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_eom = StandardScaler().fit_transform(X_eom @ T)

k_starts = list(range(2, 25))
nmi_eom = []
c_found = []

for k in k_starts:
    m = KNets(k=k, n_clusters=4)
    m.fit(X_eom)
    nmi_eom.append(NMI(y_eom, m.labels_))
    c_found.append(m.n_clusters_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(k_starts, nmi_eom, 's-', color='#2E5299', markersize=5)
ax1.set_xlabel("Starting k (k_L)")
ax1.set_ylabel("NMI")
ax1.set_title("EOM NMI vs starting k\n(target C_R = 4)")
ax1.axhline(max(nmi_eom), color='red', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

ax2.plot(k_starts, c_found, 'o-', color='#1F3864', markersize=5)
ax2.axhline(4, color='red', linestyle='--', linewidth=1.5, label='Target (4)')
ax2.set_xlabel("Starting k (k_L)")
ax2.set_ylabel("Clusters found")
ax2.set_title("EOM: clusters found vs starting k\n(should always be 4)")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle("EXP 3 — EOM stability across starting k values", fontsize=11)
plt.tight_layout()
plt.savefig(f"outputs/exp3_eom_stability.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  NMI range: [{min(nmi_eom):.3f}, {max(nmi_eom):.3f}]")
print(f"  Clusters found: always {set(c_found)}")
print("[saved] exp3_eom_stability.png")
"""
Experiments reproducing and extending the K-Nets paper:
  Maraziotis et al. (2018) Pattern Recognition
  DOI: 10.1016/j.patcog.2018.11.010

Structure
---------
EXP 1  — Synthetic datasets (approximates Fig. 5)
EXP 2  — NOM landmark sweep (approximates Fig. S1 / Section 3.1 analysis)
EXP 3  — EOM cluster recovery across k values (Section 3.1)
EXP 4  — NMI on real datasets, Table 1 style
EXP 5  — ASE comparison KNets vs KMeans across resolutions (Fig. 6 style)
EXP 6  — Assignment phase iterations analysis (Fig. 7 style)
EXP 7  — Two-layer serial with geodesic on non-linear data (Fig. 5 V/VI)
EXP 8  — Parallel two-layer speedup (Section 3.1 timing claim)

Datasets used (all offline, no network required):
  Synthetic: make_blobs, make_moons, make_circles, custom arrangements
  Real:      sklearn digits (proxy for MNIST/pen-digits/USPS)
             sklearn iris, wine, breast_cancer
"""

import sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from sklearn.datasets import (load_digits, load_iris, load_wine,
                               load_breast_cancer, make_blobs,
                               make_moons, make_circles)
from sklearn.cluster import (KMeans, AffinityPropagation, DBSCAN,
                              MeanShift, AgglomerativeClustering)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (normalized_mutual_info_score as NMI,
                              adjusted_rand_score as ARI,
                              silhouette_score)

sys.path.insert(0, '/home/claude')
from src.knets import KNets
from src.multilayer_knets import SerialTwoLayerKNets, ParallelTwoLayerKNets

warnings.filterwarnings('ignore')
SEED = 42
rng = np.random.default_rng(SEED)
OUT = 'outputs/'


# ── helpers ──────────────────────────────────────────────────────────────────

def ase(X, labels, centers):
    """Average Squared Error: mean squared distance of each point to its center."""
    total = 0.0
    for c in range(len(centers)):
        members = X[labels == c]
        if len(members):
            total += np.sum((members - centers[c]) ** 2)
    return total / len(X)

def run_kmeans(X, n_clusters, n_init=10):
    m = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=SEED)
    m.fit(X)
    return m.labels_, m.cluster_centers_

def run_kcenters(X, n_clusters):
    n_samples = X.shape[0]
    centers_idx = [0]

    min_dists = np.linalg.norm(X - X[centers_idx[0]], axis=1)
    
    for _ in range(1, n_clusters):
        new_center_idx = np.argmax(min_dists)
        centers_idx.append(new_center_idx)
      
        new_dists = np.linalg.norm(X - X[new_center_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    
    centers = X[centers_idx]
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
    return labels, centers

def run_ap(X, max_iter=200):
    m = AffinityPropagation(random_state=SEED, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X)
    return m.labels_

def run_dbscan(X, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def run_meanshift(X):
    m = MeanShift()
    m.fit(X)
    return m.labels_

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsection(title):
    print(f"\n--- {title} ---")

PALETTE = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
           '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
           '#dcbeff','#9A6324','#fffac8','#800000','#aaffc3']

def cluster_cmap(n):
    return ListedColormap(PALETTE[:n])


# ── EXP 1: Synthetic datasets — visual comparison (Fig. 5 proxy) ─────────────

section("EXP 1 — Synthetic datasets: visual clustering comparison (Fig. 5)")

datasets = []

# (II) Isotropic blobs — 31 clusters
X_blobs, y_blobs = make_blobs(n_samples=600, centers=31, cluster_std=0.4,
                               random_state=SEED)
datasets.append(("Isotropic blobs\n(31 clusters)", X_blobs, y_blobs, 31))

# (III) Non-isotropic blobs
X_aniso, y_aniso = make_blobs(n_samples=400, centers=4, random_state=SEED)
T = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = X_aniso @ T
datasets.append(("Non-isotropic blobs\n(4 clusters)", X_aniso, y_aniso, 4))

# (IV) Variable density blobs
X_var, y_var = make_blobs(n_samples=[100, 200, 400], centers=[[0,0],[3,2],[6,0]], random_state=SEED)
datasets.append(("Variable-density blobs\n(3 clusters)", X_var, y_var, 3))

# (V) Non-linear: moons
X_moons, y_moons = make_moons(n_samples=400, noise=0.07, random_state=SEED)
datasets.append(("Moons\n(2 clusters)", X_moons, y_moons, 2))

# (VI) Non-linear: circles
X_circ, y_circ = make_circles(n_samples=400, noise=0.05, factor=0.5,
                               random_state=SEED)
datasets.append(("Circles\n(2 clusters)", X_circ, y_circ, 2))

algorithms = [
    ("K-Nets EOM",   None),
    ("K-Means",      None),
    ("AP",           None),
    ("DBSCAN",       None),
    ("Mean Shift",   None),
]

k_vals    = [8,   5,   5,  5,  5]   # k for KNets per dataset
eps_vals  = [0.4, 0.6, 1.5, 0.3, 0.25]

fig, axes = plt.subplots(len(datasets), len(algorithms),
                         figsize=(18, len(datasets)*3.2))
fig.suptitle("EXP 1 — Synthetic datasets: clustering comparison\n"
             "(approximates Figure 5 of the paper)", fontsize=13, y=1.01)

for di, (name, X, y_true, nc) in enumerate(datasets):
    Xs = StandardScaler().fit_transform(X)
    subsection(name.replace('\n', ' '))
    k = k_vals[di]
    eps = eps_vals[di]

    # KNets EOM
    t0 = time.time()
    km = KNets(k=k, n_clusters=nc)
    labels_kn = km.fit_predict(Xs)
    t_kn = time.time() - t0
    nmi_kn = NMI(y_true, labels_kn)
    print(f"  KNets EOM   | C={km.n_clusters_:3d} | NMI={nmi_kn:.3f} | t={t_kn:.2f}s")

    # K-Means
    t0 = time.time()
    labels_km, centers_km = run_kmeans(Xs, nc)
    t_km = time.time() - t0
    print(f"  K-Means     | C={nc:3d} | NMI={NMI(y_true, labels_km):.3f} | t={t_km:.2f}s")

    # AP
    t0 = time.time()
    labels_ap = run_ap(Xs)
    t_ap = time.time() - t0
    nc_ap = len(np.unique(labels_ap))
    print(f"  AP          | C={nc_ap:3d} | NMI={NMI(y_true, labels_ap):.3f} | t={t_ap:.2f}s")

    # DBSCAN
    t0 = time.time()
    labels_db = run_dbscan(Xs, eps=eps)
    t_db = time.time() - t0
    nc_db = len(np.unique(labels_db[labels_db >= 0]))
    print(f"  DBSCAN      | C={nc_db:3d} | NMI={NMI(y_true, labels_db):.3f} | t={t_db:.2f}s")

    # MeanShift
    t0 = time.time()
    labels_ms = run_meanshift(Xs)
    t_ms = time.time() - t0
    nc_ms = len(np.unique(labels_ms))
    print(f"  MeanShift   | C={nc_ms:3d} | NMI={NMI(y_true, labels_ms):.3f} | t={t_ms:.2f}s")

    all_labels = [labels_kn, labels_km, labels_ap, labels_db, labels_ms]
    algo_names = ["K-Nets EOM", "K-Means", "AP", "DBSCAN", "MeanShift"]

    for ai, (alg_name, lbls) in enumerate(zip(algo_names, all_labels)):
        ax = axes[di][ai]
        nc_plot = len(np.unique(lbls[lbls >= 0]))
        cmap = plt.cm.get_cmap('tab20', max(nc_plot, 2))
        ax.scatter(Xs[:, 0], Xs[:, 1], c=lbls, cmap=cmap, s=8, linewidths=0)
        if di == 0:
            ax.set_title(alg_name, fontsize=10, fontweight='bold')
        if ai == 0:
            ax.set_ylabel(name, fontsize=8)
        nmi_v = NMI(y_true, lbls)
        ax.set_xlabel(f"NMI={nmi_v:.2f} C={len(np.unique(lbls[lbls>=0]))}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUT}/exp1_synthetic_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] exp1_synthetic_comparison.png")


# ── EXP 2: NOM landmark sweep (Section 3.1 / Fig S1) ─────────────────────────

section("EXP 2 — NOM landmark sweep: cluster count vs k (Fig. S1 / Section 3.1)")

X_sweep, y_sweep = make_blobs(n_samples=600, centers=31, cluster_std=0.4,
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

# Annotate the plateau where true clusters are recovered
true_ks = [k for k, c in zip(k_range, n_clusters_nom) if c == 31]
if true_ks:
    ax.axvspan(min(true_ks), max(true_ks), alpha=0.15, color='green',
               label=f'k range recovering 31 clusters: {min(true_ks)}–{max(true_ks)}')
    ax.legend()
    print(f"  True cluster count (31) recovered for k in [{min(true_ks)}, {max(true_ks)}]")

# Landmark percentage
n_possible = len(X_sweep)
unique_counts = len(set(n_clusters_nom))
pct = 100 * unique_counts / n_possible
print(f"  Unique landmark partitions: {unique_counts} / {n_possible} = {pct:.1f}%")
ax.text(0.97, 0.95,
        f"Unique landmark partitions:\n{unique_counts}/{n_possible} = {pct:.1f}%",
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUT}/exp2_nom_landmark_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print("[saved] exp2_nom_landmark_sweep.png")


# ── EXP 3: EOM recovery across starting k values (Section 3.1) ───────────────

section("EXP 3 — EOM cluster recovery: same result from different starting k")

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
plt.savefig(f"{OUT}/exp3_eom_stability.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  NMI range: [{min(nmi_eom):.3f}, {max(nmi_eom):.3f}]")
print(f"  Clusters found: always {set(c_found)}")
print("[saved] exp3_eom_stability.png")


# ── EXP 4: NMI on real datasets — Table 1 style ──────────────────────────────

section("EXP 4 — NMI comparison on real datasets (Table 1 proxy)")

real_datasets = []

# Digits 5-9 (proxy for MNIST 5-9)
d = load_digits()
mask = np.isin(d.target, [5,6,7,8,9])
X_d59, y_d59 = d.data[mask], d.target[mask]
real_datasets.append(("Digits 5-9\n(proxy MNIST 5-9)", X_d59, y_d59, 5))

# Digits full (proxy for pen-digits / USPS)
real_datasets.append(("Digits 0-9\n(proxy pen-digits)", d.data, d.target, 10))

# Wine
dw = load_wine()
real_datasets.append(("Wine\n(3 classes)", dw.data, dw.target, 3))

# Iris
di = load_iris()
real_datasets.append(("Iris\n(3 classes)", di.data, di.target, 3))

# Breast cancer (2 classes)
db = load_breast_cancer()
real_datasets.append(("Breast Cancer\n(2 classes)", db.data, db.target, 2))

results_table = []

for ds_name, X, y_true, nc in real_datasets:
    Xs = StandardScaler().fit_transform(X)
    subsection(ds_name.replace('\n', ' '))
    row = {"Dataset": ds_name.replace('\n', ' ')}

    # KNets EOM (Euclidean)
    t0 = time.time()
    m = KNets(k=max(5, nc*2), n_clusters=nc)
    m.fit(Xs)
    row["KNets EOM"] = NMI(y_true, m.labels_)
    row["KNets_t"] = time.time() - t0
    print(f"  KNets EOM   NMI={row['KNets EOM']:.3f}  t={row['KNets_t']:.2f}s")

    # KNets NOM (no target)
    m_nom = KNets(k=max(5, nc*2))
    m_nom.fit(Xs)
    row["KNets NOM"] = NMI(y_true, m_nom.labels_)
    print(f"  KNets NOM   NMI={row['KNets NOM']:.3f}  C={m_nom.n_clusters_}")

    # K-Means (10 inits, best)
    t0 = time.time()
    lbls_km, _ = run_kmeans(Xs, nc, n_init=10)
    row["K-Means"] = NMI(y_true, lbls_km)
    row["KMeans_t"] = time.time() - t0
    print(f"  K-Means     NMI={row['K-Means']:.3f}  t={row['KMeans_t']:.2f}s")

    # AP
    t0 = time.time()
    lbls_ap = run_ap(Xs)
    row["AP"] = NMI(y_true, lbls_ap)
    row["AP_t"] = time.time() - t0
    print(f"  AP          NMI={row['AP']:.3f}  C={len(np.unique(lbls_ap))}  t={row['AP_t']:.2f}s")

    # Agglomerative (Ward) — added as extra comparison
    lbls_agg = AgglomerativeClustering(n_clusters=nc).fit_predict(Xs)
    row["Agglomerative"] = NMI(y_true, lbls_agg)
    print(f"  Agglom.     NMI={row['Agglomerative']:.3f}")

    results_table.append(row)

# Plot Table 1 as heatmap-style bar chart
fig, ax = plt.subplots(figsize=(12, 5))
ds_names = [r["Dataset"] for r in results_table]
algos = ["KNets EOM", "KNets NOM", "K-Means", "AP", "Agglomerative"]
colors = ['#1F3864', '#2E5299', '#e6194b', '#f58231', '#3cb44b']
x = np.arange(len(ds_names))
width = 0.15

for i, (algo, color) in enumerate(zip(algos, colors)):
    vals = [r.get(algo, 0) for r in results_table]
    bars = ax.bar(x + i*width - width*2, vals, width, label=algo,
                  color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=9)
ax.set_ylabel("NMI")
ax.set_ylim(0, 1.15)
ax.set_title("EXP 4 — NMI comparison on real datasets (Table 1 proxy)\n"
             "Higher is better")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUT}/exp4_nmi_real_datasets.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] exp4_nmi_real_datasets.png")


# ── EXP 5: ASE comparison KNets vs KMeans across resolutions (Fig. 6) ────────

section("EXP 5 — ASE: K-Nets vs K-Means across cluster counts (Fig. 6 style)")

d = load_digits()
X_ase = StandardScaler().fit_transform(d.data)
y_ase = d.target

cluster_range = list(range(5, 51, 5))
ase_knets, ase_kmeans, nmi_knets, nmi_kmeans = [], [], [], []
time_knets, time_kmeans = [], []

print("Computing ASE and NMI across cluster counts on digits dataset...")
for nc in cluster_range:
    print(f"  nc={nc}...")

    t0 = time.time()
    m = KNets(k=max(5, nc), n_clusters=nc)
    m.fit(X_ase)
    time_knets.append(time.time() - t0)
    ase_knets.append(ase(X_ase, m.labels_, m.cluster_centers_))
    nmi_knets.append(NMI(y_ase, m.labels_))

    t0 = time.time()
    lbls_km, ctrs_km = run_kmeans(X_ase, nc, n_init=10)
    time_kmeans.append(time.time() - t0)
    ase_kmeans.append(ase(X_ase, lbls_km, ctrs_km))
    nmi_kmeans.append(NMI(y_ase, lbls_km))

ase_knets, ase_kmeans = np.array(ase_knets), np.array(ase_kmeans)
pct_improvement = 100 * (ase_kmeans - ase_knets) / ase_kmeans

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(cluster_range, ase_knets, 'o-', color='#1F3864', label='K-Nets EOM')
axes[0].plot(cluster_range, ase_kmeans, 's--', color='#e6194b', label='K-Means')
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("ASE")
axes[0].set_title("ASE vs cluster count")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].bar(cluster_range, pct_improvement, color='#2E5299', alpha=0.8, width=3.5)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xlabel("Number of clusters")
axes[1].set_ylabel("% ASE improvement of K-Nets over K-Means")
axes[1].set_title("K-Nets ASE improvement over K-Means\n(positive = K-Nets better)")
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].plot(cluster_range, nmi_knets, 'o-', color='#1F3864', label='K-Nets EOM')
axes[2].plot(cluster_range, nmi_kmeans, 's--', color='#e6194b', label='K-Means')
axes[2].set_xlabel("Number of clusters")
axes[2].set_ylabel("NMI")
axes[2].set_title("NMI vs cluster count (digits, 10 true classes)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle("EXP 5 — ASE and NMI: K-Nets vs K-Means (Fig. 6 style)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/exp5_ase_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Mean ASE improvement of K-Nets over K-Means: {pct_improvement.mean():.1f}%")
print("[saved] exp5_ase_comparison.png")


# ── EXP 6: Assignment phase iterations analysis (Fig. 7) ─────────────────────

section("EXP 6 — Assignment phase iterations: 1-iter vs full convergence (Fig. 7)")

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

ase_1iter, ase_full = np.array(ase_1iter), np.array(ase_full)
pct_diff = 100 * (ase_1iter - ase_full) / ase_full

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(cluster_range_it, ase_full, 'o-', color='#1F3864', label='Full convergence')
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
plt.savefig(f"{OUT}/exp6_iterations_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Max % ASE degradation with 1 iter: {pct_diff.max():.2f}%")
print(f"  Mean iterations to convergence: {np.mean(iters_used):.1f}")
print("[saved] exp6_iterations_analysis.png")


# ── EXP 7: Two-layer serial with geodesic on non-linear data (Fig. 5 V/VI) ───

section("EXP 7 — Two-layer serial + geodesic on non-linear datasets (Fig. 5 V/VI)")

nonlinear_sets = [
    ("Moons", *make_moons(n_samples=500, noise=0.06, random_state=SEED), 2),
    ("Circles", *make_circles(n_samples=500, noise=0.04, factor=0.45, random_state=SEED), 2),
]

fig, axes = plt.subplots(len(nonlinear_sets), 5,
                         figsize=(18, len(nonlinear_sets)*3.5))
fig.suptitle("EXP 7 — Non-linear datasets: single-layer vs two-layer serial K-Nets + geodesic\n"
             "(approximates Fig. 5 V and VI)", fontsize=11, y=1.02)

for di, (name, X, y_true, nc) in enumerate(nonlinear_sets):
    Xs = StandardScaler().fit_transform(X)
    subsection(name)

    # Single-layer K-Nets EOM (Euclidean)
    m_single = KNets(k=5, n_clusters=nc)
    lbls_single = m_single.fit_predict(Xs)
    nmi_single = NMI(y_true, lbls_single)
    print(f"  Single KNets EOM   NMI={nmi_single:.3f}")

    # Two-layer serial with geodesic
    nmi_serial_best = 0
    lbls_serial_best = None
    for gn in [5, 8, 12]:
        try:
            m_ser = SerialTwoLayerKNets(k1=3, k2=3, n_clusters=nc, geo_neighbors=gn)
            lbls_ser = m_ser.fit_predict(Xs)
            nmi_ser = NMI(y_true, lbls_ser)
            print(f"  Serial geo_n={gn:2d}      NMI={nmi_ser:.3f} | L1 exemplars={m_ser.layer1_.n_clusters_}")
            if nmi_ser > nmi_serial_best:
                nmi_serial_best = nmi_ser
                lbls_serial_best = lbls_ser
        except Exception as e:
            print(f"  Serial geo_n={gn} failed: {e}")

    # K-Means
    lbls_km, _ = run_kmeans(Xs, nc)
    nmi_km = NMI(y_true, lbls_km)
    print(f"  K-Means            NMI={nmi_km:.3f}")

    # AP
    lbls_ap = run_ap(Xs)
    nmi_ap = NMI(y_true, lbls_ap)
    print(f"  AP                 NMI={nmi_ap:.3f} C={len(np.unique(lbls_ap))}")

    # DBSCAN (tuned per dataset)
    eps_nl = 0.25 if name == "Moons" else 0.2
    lbls_db = run_dbscan(Xs, eps=eps_nl, min_samples=5)
    nmi_db = NMI(y_true, lbls_db)
    print(f"  DBSCAN             NMI={nmi_db:.3f} C={len(np.unique(lbls_db[lbls_db>=0]))}")

    plot_data = [
        ("K-Nets\nSingle-layer", lbls_single, nmi_single),
        ("K-Nets\nSerial+Geodesic", lbls_serial_best if lbls_serial_best is not None else lbls_single, nmi_serial_best),
        ("K-Means", lbls_km, nmi_km),
        ("AP", lbls_ap, nmi_ap),
        ("DBSCAN", lbls_db, nmi_db),
    ]

    for ai, (alg_name, lbls, nmi_v) in enumerate(plot_data):
        ax = axes[di][ai] if len(nonlinear_sets) > 1 else axes[ai]
        cmap = plt.cm.get_cmap('tab10', max(len(np.unique(lbls[lbls>=0])), 2))
        ax.scatter(Xs[:, 0], Xs[:, 1], c=lbls, cmap=cmap, s=8, linewidths=0)
        if di == 0:
            ax.set_title(alg_name, fontsize=9, fontweight='bold')
        if ai == 0:
            ax.set_ylabel(name, fontsize=9)
        ax.set_xlabel(f"NMI={nmi_v:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUT}/exp7_nonlinear_geodesic.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] exp7_nonlinear_geodesic.png")


# ── EXP 8: Parallel two-layer speedup (Section 3.1 timing claim) ─────────────

section("EXP 8 — Parallel two-layer K-Nets: speedup vs single-layer (Section 3.1)")

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

    print(f"    Single  t={t_single[-1]:.3f}s  NMI={nmi_single_p[-1]:.3f}")
    print(f"    Parallel t={t_parallel[-1]:.3f}s NMI={nmi_parallel_p[-1]:.3f}")

speedup = [ts/tp for ts, tp in zip(t_single, t_parallel)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

ax1.plot(sizes, t_single, 'o-', color='#e6194b', label='Single-layer')
ax1.plot(sizes, t_parallel, 's-', color='#1F3864', label='Parallel two-layer')
ax1.set_xlabel("Dataset size (N)")
ax1.set_ylabel("Time (s)")
ax1.set_title("Runtime: single vs parallel two-layer")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(sizes, speedup, color='#3cb44b', alpha=0.85, width=sizes[0]*0.6)
ax2.axhline(1, color='black', linewidth=0.8)
ax2.set_xlabel("Dataset size (N)")
ax2.set_ylabel("Speedup factor (single/parallel)")
ax2.set_title("Speedup factor of parallel two-layer\nvs single-layer")
ax2.grid(True, alpha=0.3, axis='y')

ax3.plot(sizes, nmi_single_p, 'o-', color='#e6194b', label='Single-layer')
ax3.plot(sizes, nmi_parallel_p, 's-', color='#1F3864', label='Parallel two-layer')
ax3.set_xlabel("Dataset size (N)")
ax3.set_ylabel("NMI")
ax3.set_title("NMI: single vs parallel two-layer\n(quality preservation check)")
ax3.legend(); ax3.grid(True, alpha=0.3)

fig.suptitle("EXP 8 — Parallel two-layer K-Nets: speedup and quality (Section 3.1)",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/exp8_parallel_speedup.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Max speedup: {max(speedup):.2f}x at n={sizes[np.argmax(speedup)]}")
print("[saved] exp8_parallel_speedup.png")


# ── SUMMARY ──────────────────────────────────────────────────────────────────

section("ALL EXPERIMENTS COMPLETE")
print(f"""
Output files written to {OUT}/:
  exp1_synthetic_comparison.png   — Fig. 5: visual clustering on 5 synthetic datasets
  exp2_nom_landmark_sweep.png     — Fig. S1: cluster count vs k, landmark %
  exp3_eom_stability.png          — Section 3.1: EOM stability across starting k
  exp4_nmi_real_datasets.png      — Table 1: NMI on 5 real datasets
  exp5_ase_comparison.png         — Fig. 6: ASE + NMI vs cluster count
  exp6_iterations_analysis.png    — Fig. 7: 1-iter vs full convergence
  exp7_nonlinear_geodesic.png     — Fig. 5 V/VI: serial+geodesic on non-linear data
  exp8_parallel_speedup.png       — Section 3.1: parallel two-layer speedup
""")
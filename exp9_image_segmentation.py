import time
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, transform
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

from src.knets import KNets

TARGET_H, TARGET_W = 321, 481

def load_image(name):
    img_rgb = getattr(data, name)()
    img_rgb = transform.resize(
        img_rgb, (TARGET_H, TARGET_W),
        anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)
    img_lab = color.rgb2lab(img_rgb)
    pixels = img_lab.reshape(-1, 3).astype(float)
    return img_rgb, img_lab, pixels

images = [
    ("astronaut", "Astronaut"),
    ("chelsea", "Cat (Chelsea)"),
    ("coffee", "Coffee"),
    ("rocket", "Rocket"),
]

N_CLUSTERS = 2

def segment_knets(pixels):
    t0 = time.time()
    n_subsets = max(50, len(pixels) // 500)
    m = ParallelTwoLayerKNets(
        k1=10, k2=8, n_clusters=N_CLUSTERS, n_subsets=n_subsets
    )
    labels = m.fit_predict(pixels)
    print(f"    [K-Nets] done in {time.time()-t0:.1f}s | cluster sizes: {np.bincount(labels)}")
    return labels

def segment_kmeans(pixels):
    t0 = time.time()
    m = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    labels = m.fit_predict(pixels)
    print(f"    [K-Means] done in {time.time()-t0:.1f}s | cluster sizes: {np.bincount(labels)}")
    return labels

def segment_meanshift(pixels):
    t0 = time.time()
    sub = pixels[np.random.default_rng(42).choice(len(pixels), 2000, replace=False)]
    bw = estimate_bandwidth(sub, quantile=0.15, n_samples=500)
    bw = max(bw, 5.0)
    m = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=1)
    labels = m.fit_predict(pixels)
    n_found = len(np.unique(labels))
    print(f"    [MeanShift] done in {time.time()-t0:.1f}s | clusters found: {n_found}")
    if n_found > N_CLUSTERS:
        order = np.argsort(-np.bincount(labels))
        remap = np.zeros(n_found, dtype=int)
        remap[order[:N_CLUSTERS]] = np.arange(N_CLUSTERS)
        remap[order[N_CLUSTERS:]] = N_CLUSTERS - 1
        labels = remap[labels]
    return labels

def foreground_cluster(img_rgb, labels, h, w):
    img_hsv = color.rgb2hsv(img_rgb)
    sat = img_hsv[:, :, 1].ravel()
    cx, cy = w / 2, h / 2
    xs = np.tile(np.arange(w), h)
    ys = np.repeat(np.arange(h), w)

    n_c = len(np.unique(labels))
    scores = []
    for c in range(n_c):
        mask = labels == c
        mean_sat = sat[mask].mean()
        dist_center = np.sqrt(((xs[mask] - cx)**2 + (ys[mask] - cy)**2)).mean()
        norm_dist = dist_center / np.sqrt(cx**2 + cy**2)
        score = mean_sat - 0.3 * norm_dist
        scores.append(score)
    return int(np.argmax(scores))

print("=" * 65)
print("  Image Segmentation — Figure 10 reproduction")
print("=" * 65)

fig, axes = plt.subplots(len(images), 4, figsize=(16, len(images) * 4.2))

col_titles = ["Original", "K-Nets", "K-Means", "MeanShift"]

results = []

for ri, (img_name, img_label) in enumerate(images):
    print(f"\n--- {img_label} ---")
    img_rgb, img_lab, pixels = load_image(img_name)
    h, w = TARGET_H, TARGET_W

    row_result = {"name": img_label}
    all_labels = {}

    all_labels["K-Nets"] = segment_knets(pixels)
    all_labels["K-Means"] = segment_kmeans(pixels)
    all_labels["MeanShift"] = segment_meanshift(pixels)

    for algo_name, lbls in all_labels.items():
        n_found = len(np.unique(lbls))
        row_result[f"{algo_name}_n"] = n_found

    ax = axes[ri][0]
    ax.imshow(img_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    if ri == 0:
        ax.set_title(col_titles[0])
    ax.set_ylabel(img_label)

    for ai, (algo_name, lbls) in enumerate(all_labels.items()):
        ax = axes[ri][ai + 1]

        fg = foreground_cluster(img_rgb, lbls, h, w)
        fg_mask = (lbls == fg).reshape(h, w)

        display = img_rgb.copy().astype(float)
        bg_mask = ~fg_mask
        display[bg_mask] = display[bg_mask] * 0.15

        ax.imshow(display.astype(np.uint8))
        ax.set_xticks([]); ax.set_yticks([])

        n_fg = fg_mask.sum()
        pct = 100 * n_fg / (h * w)
        ax.set_xlabel(f"{pct:.1f}%")

        if ri == 0:
            ax.set_title(col_titles[ai + 1])

        row_result[f"{algo_name}_fg_pct"] = pct

    results.append(row_result)
    print("  Done.")

plt.tight_layout()
out_path = "outputs/exp9_image_segmentation.png"
plt.savefig(out_path, dpi=130, bbox_inches='tight')
plt.close()

print(f"\n[saved] {out_path}")

print("\n" + "=" * 65)
print("  Summary")
print("=" * 65)
print(f"{'Image':<20} {'K-Nets':>10} {'K-Means':>10} {'MeanShift':>10}")
print("-" * 52)
for r in results:
    print(f"{r['name']:<20} "
          f"{r.get('K-Nets_fg_pct', 0):>9.1f}% "
          f"{r.get('K-Means_fg_pct', 0):>9.1f}% "
          f"{r.get('MeanShift_fg_pct', 0):>9.1f}%")
"""
OPTIONAL (desirable item) - Unsupervised learning with k-means.

    python explore_kmeans.py

Clusters ALL your crops by their colour features WITHOUT using the labels,
then checks how well the clusters match the labels you gave them.
  - results/kmeans_table.txt : how many images of each label fell in each cluster
  - results/kmeans_pca.png   : the features squashed to 2-D (PCA), coloured by
                               your label (left) and by k-means cluster (right)
Discuss: which colours separate on their own? Which get mixed up, and why?
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import config as cfg
from progress import print_progress
from colour_model import extract, list_dataset, load_images


def main():
    paths, labels = list_dataset()
    if not paths:
        raise SystemExit("No images in dataset/.")
    X = StandardScaler().fit_transform(extract(load_images(paths), "colour"))
    k = len(cfg.CLASSES)
    clusters = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)

    lines = [f"k = {k}   adjusted Rand index = {adjusted_rand_score(labels, clusters):.3f} "
             f"(1 = clusters match your labels exactly, 0 = no better than chance)", ""]
    for c in range(k):
        counts = Counter(l for l, cl in zip(labels, clusters) if cl == c)
        lines.append(f"cluster {c}: " + ", ".join(f"{l} {n}" for l, n in counts.most_common()))
    text = "\n".join(lines)
    print(text)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    (cfg.RESULTS_DIR / "kmeans_table.txt").write_text(text + "\n")

    xy = PCA(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in cfg.CLASSES:
        sel = np.array(labels) == label
        axes[0].scatter(xy[sel, 0], xy[sel, 1], s=8, label=label)
    axes[0].set_title("Your labels")
    axes[0].legend()
    axes[1].scatter(xy[:, 0], xy[:, 1], s=8, c=clusters, cmap="tab10")
    axes[1].set_title("k-means clusters (no labels used)")
    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / "kmeans_pca.png", dpi=150)
    print(f"Saved {cfg.RESULTS_DIR / 'kmeans_pca.png'}")


if __name__ == "__main__":
    print_progress(None)
    main()
    print_progress(None)

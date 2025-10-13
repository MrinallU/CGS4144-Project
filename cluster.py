#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency

# ========== CONFIG ==========
EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
RESULTS_DIR = "./results"

METHOD = "kmeans"
N_CLUSTERS = 4
GENE_COUNTS = [10, 100, 1000, 10000]

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_clustering(X, method="kmeans", n_clusters=4):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
        labels = model.fit_predict(X)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        labels = model.fit_predict(X)
    elif method == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors", random_state=0
        )
        labels = model.fit_predict(X)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return labels


def plot_pca_clusters(X_scaled, labels, n_genes, method, outdir):
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    for cl in np.unique(labels):
        sel = labels == cl
        plt.scatter(pcs[sel, 0], pcs[sel, 1], label=f"Cluster {cl}", alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title(f"{method.upper()} clustering ({n_genes} genes)")
    plt.legend(frameon=False)
    plt.tight_layout()
    outpath = os.path.join(outdir, f"{method}_{n_genes}genes_pca.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[OK] Saved PCA plot for {n_genes} genes → {outpath}")


def chi2_cluster_comparison(labels_dict):
    results = []
    keys = list(labels_dict.keys())
    for i, j in itertools.combinations(keys, 2):
        a, b = labels_dict[i], labels_dict[j]
        contingency = pd.crosstab(a, b)
        chi2, p, dof, _ = chi2_contingency(contingency)
        results.append({
            "Comparison": f"{i} vs {j}",
            "Chi2": chi2,
            "df": dof,
            "p_value": p,
        })
    return pd.DataFrame(results)


def main():
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t").set_index("refinebio_accession_code")

    common = expr.columns.intersection(meta.index)
    expr = expr[common]
    meta = meta.loc[common]

    log_expr = np.log1p(expr)

    results_labels = {}

    for n_genes in GENE_COUNTS:
        variances = log_expr.var(axis=1)
        top_genes = variances.sort_values(ascending=False).head(n_genes).index
        expr_subset = log_expr.loc[top_genes]

        X = expr_subset.T
        X_scaled = StandardScaler().fit_transform(X)

        labels = run_clustering(X_scaled, method=METHOD, n_clusters=N_CLUSTERS)
        results_labels[n_genes] = labels

        meta[f"cluster_{n_genes}"] = labels
        plot_pca_clusters(X_scaled, labels, n_genes, METHOD, RESULTS_DIR)

    meta.to_csv(os.path.join(RESULTS_DIR, f"{METHOD}_clusters_all_sizes.tsv"), sep="\t")

    chi_df = chi2_cluster_comparison(results_labels)
    chi_path = os.path.join(RESULTS_DIR, f"{METHOD}_chi2_comparison.tsv")
    chi_df.to_csv(chi_path, sep="\t", index=False)
    print(f"[OK] Saved chi-squared comparison table → {chi_path}")
    print(chi_df)


if __name__ == "__main__":
    main()

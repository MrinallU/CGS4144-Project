#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation


# ================================
# Config
# ================================
EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
RESULTS_DIR = "./results"
KMEANS_LABELS = os.path.join(RESULTS_DIR, "kmeans_labels.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "heatmap_kmeans.png")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================
# Helpers
# ================================
def derive_group(title: str) -> str:
    """Map refine.bio titles to Polyp/Cancer categories."""
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"


# ================================
# Main
# ================================
def main():
    # ---------- Load expression + metadata ----------
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t").set_index("refinebio_accession_code")

    # Align samples
    common = expr.columns.intersection(meta.index)
    expr = np.log1p(expr.loc[:, common])
    meta = meta.loc[common]
    print(f"Expression matrix: {expr.shape}")

    # ---------- Subset to 5000 most variable genes ----------
    top_genes = expr.var(axis=1).nlargest(5000).index
    expr_top = expr.loc[top_genes]

    # ---------- KMeans clustering ----------
    if os.path.exists(KMEANS_LABELS):
        kmeans_labels = pd.read_csv(KMEANS_LABELS, index_col=0).iloc[:, 0]
        print(f"Loaded existing KMeans labels from {KMEANS_LABELS}")
    else:
        print("No KMeans labels found — running KMeans (k=4)...")
        X = expr_top.T
        kmeans = KMeans(n_clusters=4, random_state=0)
        labels = kmeans.fit_predict(X)
        kmeans_labels = pd.Series(labels, index=X.index, name="cluster")
        kmeans_labels.to_csv(KMEANS_LABELS)
        print(f"[OK] Saved new KMeans labels → {KMEANS_LABELS}")

    kmeans_labels = kmeans_labels.reindex(expr_top.columns)

    # ---------- Biological groups ----------
    meta["group"] = meta["refinebio_title"].apply(derive_group)

    # ---------- Annotation setup ----------
    annotations = pd.DataFrame(
        {"KMeans Cluster": kmeans_labels.astype(str), "Group": meta["group"]},
        index=expr_top.columns,
    )

    colors = {
        "Group": {"Cancer": "red", "Polyp": "blue", "Other": "gray"},
        "KMeans Cluster": {
            str(lbl): matplotlib.colors.rgb2hex(plt.get_cmap("tab10")(i % 10))
            for i, lbl in enumerate(
                sorted(annotations["KMeans Cluster"].dropna().unique())
            )
        },
    }

    # ---------- Z-score normalize expression ----------
    z_expr = (expr_top - expr_top.mean(axis=1).values[:, None]) / expr_top.std(
        axis=1
    ).values[:, None]

    # ---------- Heatmap annotation ----------
    anno = HeatmapAnnotation(df=annotations, colors=colors, legend=True)

    # ---------- Plot heatmap ----------
    print("Plotting KMeans heatmap with biological group annotation...")

    try:
        cmp = ClusterMapPlotter(
            data=z_expr,
            top_annotation=anno,  # modern argument name
            cmap="RdBu_r",
            row_cluster=True,
            col_cluster=True,
            show_rownames=False,
            show_colnames=False,
        )
    except TypeError:
        cmp = ClusterMapPlotter(
            data=z_expr,
            col_annotations=anno,  # fallback for older versions
            cmap="RdBu_r",
            row_cluster=True,
            col_cluster=True,
            show_rownames=False,
            show_colnames=False,
        )

    cmp.plot()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved heatmap → {OUT_PNG}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, mygene

EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
SIGNATURE_PATH = "./results/svm_signature.json"
KMEANS_PATH = "./results/kmeans_labels.csv"
HEATMAP_OUT = "./results/svm_heatmap.png"


def map_ensembl_to_symbol(genes):
    mg = mygene.MyGeneInfo()
    q = mg.querymany(
        genes,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
    )
    if isinstance(q, pd.DataFrame) and "symbol" in q.columns:
        return q["symbol"].dropna().to_dict()
    return {}


def main():
    os.makedirs("./results", exist_ok=True)

    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t").set_index("refinebio_accession_code")

    def derive_group(title):
        t = str(title).upper()
        if any(k in t for k in ["SSA", "HP", "AP"]):
            return "Polyp"
        if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
            return "Cancer"
        return "Other"

    meta["group"] = meta["refinebio_title"].apply(derive_group)

    common = expr.columns.intersection(meta.index)
    expr, meta = expr[common], meta.loc[common]
    expr_log = np.log1p(expr)

    with open(SIGNATURE_PATH) as f:
        sig = json.load(f)
    svm_genes = sig.get("svm_linear", list(sig.keys()))
    matched = [g for g in svm_genes if g in expr_log.index]

    if not matched:
        print("[WARN] No direct matches — mapping...")
        mapping = map_ensembl_to_symbol(svm_genes)
        matched = [
            mapping[g]
            for g in svm_genes
            if g in mapping and mapping[g] in expr_log.index
        ]

    if not matched:
        raise ValueError("No predictive genes found in expression matrix.")
    print(f"[INFO] Matched {len(matched)} genes")

    expr_z = expr_log.loc[matched]
    expr_z = (
        expr_z.sub(expr_z.mean(axis=1), axis=0)
        .div(expr_z.std(axis=1), axis=0)
        .fillna(0)
    )

    group_palette = {"Cancer": "#d62728", "Polyp": "#1f77b4", "Other": "#aaaaaa"}
    group_colors = meta["group"].map(group_palette)

    if os.path.exists(KMEANS_PATH):
        km = pd.read_csv(
            KMEANS_PATH, index_col=0, header=None, names=["cluster"]
        ).reindex(expr_z.columns)
        uniq = sorted(km["cluster"].dropna().unique())
        cmap = sns.color_palette("tab10", len(uniq))
        cluster_palette = dict(zip(uniq, cmap))
        cluster_colors = km["cluster"].map(cluster_palette)
    else:
        cluster_colors = pd.Series(
            ["#cccccc"] * len(expr_z.columns), index=expr_z.columns
        )

    col_colors = pd.DataFrame({"Group": group_colors, "KMeans": cluster_colors})

    print("[INFO] Plotting Seaborn clustermap...")
    sns.set(context="paper", font_scale=0.8)
    g = sns.clustermap(
        expr_z,
        cmap="RdBu_r",
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=True,
        figsize=(10, 8),
        dendrogram_ratio=(0.1, 0.2),
        cbar_pos=(0.02, 0.8, 0.03, 0.18),
    )
    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Predictive Genes")
    plt.title("SVM Predictive Gene Signature Heatmap", pad=80)
    g.savefig(HEATMAP_OUT, dpi=300)
    print(f"[OK] Saved → {os.path.abspath(HEATMAP_OUT)}")


if __name__ == "__main__":
    main()

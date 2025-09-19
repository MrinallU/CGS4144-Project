#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================
# Imports (all at the top)
# ================================
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import mygene

from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation


# ================================
# Config
# ================================
EXPR_PATH = "./SRP068591.tsv"  # refine.bio expression matrix (genes x samples)
META_PATH = "./metadata_SRP068591.tsv"  # refine.bio metadata
RESULTS_DIR = "./results"

VOLCANO_PNG = os.path.join(RESULTS_DIR, "volcano.png")
ALL_TSV = os.path.join(RESULTS_DIR, "all_DEGs.tsv")
TOP50_TSV = os.path.join(RESULTS_DIR, "top50_DEGs.tsv")
HEATMAP_PNG = os.path.join(RESULTS_DIR, "heatmap_sig.png")

GROUP_POS = "Cancer"
GROUP_NEG = "Polyp"
LFC_THRESH = 1.0
PADJ_THRESH = 0.05
VOLCANO_LABELS = 15
RANDOM_STATE = 0


# ================================
# Helpers
# ================================
def safe_make_dir(path):
    os.makedirs(path, exist_ok=True)


def strip_ensembl_version(idx: pd.Index) -> pd.Index:
    return idx.str.replace(r"\.\d+$", "", regex=True)


def derive_group_from_title(title: str) -> str:
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"


def plot_density(gene_medians: pd.Series, out_png: str | None = None):
    plt.figure(figsize=(6, 4))
    gene_medians.plot(kind="density")
    plt.title("Per-gene median (log1p counts) density")
    plt.xlabel("Median log1p expression per gene")
    plt.ylabel("Density")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
        plt.close()
    else:
        plt.show()


def scatter_by_group(X2, labels, title, xlabel, ylabel, legend_title):
    plt.figure(figsize=(6, 5))
    for g in np.unique(labels):
        sel = labels == g
        plt.scatter(X2[sel, 0], X2[sel, 1], label=g, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, title=legend_title)
    plt.tight_layout()
    plt.show()


def volcano_plot(
    df: pd.DataFrame,
    lfc_col="log2FoldChange",
    padj_col="padj",
    lfc_thresh=1.0,
    padj_thresh=0.05,
    max_labels=15,
    outfile="results/volcano.png",
    x_label=f"log2 fold change ({GROUP_POS} vs {GROUP_NEG})",
):
    plot_df = df[[lfc_col, padj_col]].dropna().copy()
    if plot_df.empty:
        raise ValueError("Volcano: no rows with both log2FC and padj available.")

    plot_df["neglog10padj"] = -np.log10(plot_df[padj_col].clip(lower=1e-300))

    sig = (plot_df[padj_col] < padj_thresh) & (plot_df[lfc_col].abs() >= lfc_thresh)
    up = sig & (plot_df[lfc_col] > 0)
    dn = sig & (plot_df[lfc_col] < 0)

    print(
        f"Volcano input rows: {len(plot_df)} | significant: {int(sig.sum())} "
        f"(up: {int(up.sum())}, down: {int(dn.sum())})"
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    ax.scatter(
        plot_df.loc[~sig, lfc_col],
        plot_df.loc[~sig, "neglog10padj"],
        s=10,
        alpha=0.6,
        label="NS",
    )
    ax.scatter(
        plot_df.loc[dn, lfc_col],
        plot_df.loc[dn, "neglog10padj"],
        s=12,
        alpha=0.8,
        label=f"Signif down (≤ -{lfc_thresh})",
    )
    ax.scatter(
        plot_df.loc[up, lfc_col],
        plot_df.loc[up, "neglog10padj"],
        s=12,
        alpha=0.8,
        label=f"Signif up (≥ +{lfc_thresh})",
    )

    ax.axvline(+lfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, linestyle="--", linewidth=1)
    ax.axhline(-np.log10(padj_thresh), linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("Volcano plot")
    ax.legend(frameon=False)

    fig.tight_layout()
    abs_out = os.path.abspath(outfile)
    fig.savefig(abs_out, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved volcano: {abs_out}")


def plot_sig_heatmap(log_expr, res_df, meta, outfile, padj_thresh=0.05, lfc_thresh=1.0):
    sig_genes = res_df.dropna(subset=["padj", "log2FoldChange"])
    sig_genes = sig_genes[
        (sig_genes["padj"] < padj_thresh)
        & (sig_genes["log2FoldChange"].abs() >= lfc_thresh)
    ]
    print(f"Significant genes for heatmap: {sig_genes.shape[0]}")
    if sig_genes.empty:
        print("[WARN] No significant DEGs found, skipping heatmap.")
        return

    expr_sig = log_expr.loc[sig_genes.index]
    expr_z = (expr_sig - expr_sig.mean(axis=1).values[:, None]) / expr_sig.std(
        axis=1
    ).values[:, None]

    groups = meta.loc[expr_sig.columns, "group"]

    col_anno = HeatmapAnnotation(
        Group=groups.map({"Cancer": "red", "Polyp": "blue"}), legend=True
    )

    plt.figure(figsize=(10, 8))  # <-- set figure size here
    cmp = ClusterMapPlotter(
        data=expr_z,
        col_cluster=True,
        row_cluster=True,
        top_annotation=col_anno,
        cmap="RdBu_r",
    )

    cmp.plot()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved DEG heatmap: {os.path.abspath(outfile)}")


# ================================
# Main
# ================================
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    safe_make_dir(RESULTS_DIR)

    # ---------- Load data ----------
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t")
    meta = meta.set_index("refinebio_accession_code")
    common = expr.columns.intersection(meta.index)
    expr = expr.loc[:, common]
    meta = meta.loc[common]

    print(f"Matrix size (genes x samples): {expr.shape}")
    print(f"# unique genes (row index): {expr.index.nunique()}")

    # ---------- Ensembl -> HGNC symbols ----------
    ens = strip_ensembl_version(expr.index)
    mg = mygene.MyGeneInfo()
    conv = mg.querymany(
        ens.unique().tolist(),
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )
    if not isinstance(conv, pd.DataFrame):
        conv = pd.DataFrame(conv)
    symbol_map = conv["symbol"].dropna().to_dict()
    new_index = ens.map(symbol_map)
    mapped_mask = new_index.notna()
    expr_mapped = expr.loc[mapped_mask].copy()
    expr_mapped.index = new_index[mapped_mask]
    expr_mapped = expr_mapped.groupby(level=0, sort=False).median()
    print(f"After mapping/collapsing -> shape: {expr_mapped.shape}")

    # ---------- Log1p ----------
    log_expr = np.log1p(expr_mapped)
    gene_medians = log_expr.median(axis=1)
    plot_density(gene_medians)

    # ---------- Groups ----------
    meta["group"] = meta["refinebio_title"].apply(derive_group_from_title)
    meta = meta.loc[meta["group"].isin([GROUP_NEG, GROUP_POS])]
    log_expr = log_expr.loc[:, meta.index]
    print("\n[Groups]")
    print(meta["group"].value_counts())

    # ---------- PCA/t-SNE/UMAP ----------
    X = (log_expr.T - log_expr.T.mean()).values
    labels = meta.loc[log_expr.columns, "group"].astype(str).values
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pc = pca.fit_transform(X)
    scatter_by_group(
        pc,
        labels,
        "PCA of samples",
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)",
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)",
        "group",
    )
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
        perplexity=min(30, max(5, X.shape[0] // 4)),
    )
    ts = tsne.fit_transform(X)
    scatter_by_group(ts, labels, "t-SNE of samples", "t-SNE1", "t-SNE2", "group")
    um = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=RANDOM_STATE
    )
    uu = um.fit_transform(X)
    scatter_by_group(uu, labels, "UMAP of samples", "UMAP1", "UMAP2", "group")

    # ---------- DESeq2 ----------
    deseq2 = importr("DESeq2")
    try:
        importr("apeglm")
        has_apeglm = True
    except Exception:
        has_apeglm = False

    counts_df = expr_mapped.loc[:, meta.index].fillna(0).round().astype(int)

    # Build coldata directly from meta (guaranteed aligned)
    coldata = meta.loc[counts_df.columns, ["group"]].copy()
    coldata["group"] = pd.Categorical(
        coldata["group"], categories=[GROUP_NEG, GROUP_POS]
    )
    with localconverter(default_converter + pandas2ri.converter):
        r_counts = pandas2ri.py2rpy(counts_df)
        r_coldata = pandas2ri.py2rpy(coldata)
    dds = deseq2.DESeqDataSetFromMatrix(
        countData=r_counts, colData=r_coldata, design=ro.Formula("~ group")
    )
    dds = deseq2.DESeq(dds)

    res = deseq2.results(dds, contrast=ro.StrVector(["group", GROUP_POS, GROUP_NEG]))
    if has_apeglm:
        res = deseq2.lfcShrink(dds, coef=2, type="apeglm")
    r_df = ro.r["as.data.frame"](res)
    with localconverter(default_converter + pandas2ri.converter):
        res_df = pandas2ri.rpy2py(r_df).copy()

    res_df.index = counts_df.index
    res_df = res_df.sort_values("padj", na_position="last")
    res_df.to_csv(ALL_TSV, sep="\t")
    res_df.dropna(subset=["padj"]).head(50).to_csv(TOP50_TSV, sep="\t")
    print(f"[OK] Saved DE results to {RESULTS_DIR}")

    # ---------- Volcano ----------
    volcano_plot(res_df, outfile=VOLCANO_PNG)

    # ---------- Heatmap ----------
    plot_sig_heatmap(
        log_expr,
        res_df,
        meta,
        HEATMAP_PNG,
        padj_thresh=PADJ_THRESH,
        lfc_thresh=LFC_THRESH,
    )


if __name__ == "__main__":
    main()

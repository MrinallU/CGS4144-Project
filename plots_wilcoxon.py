#!/usr/bin/env python3
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation

# ----------------- PATHS & PARAMETERS -----------------
EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
RESULTS_DIR = "./results"

VOLCANO_PNG = os.path.join(RESULTS_DIR, "volcano.png")
ALL_TSV = os.path.join(RESULTS_DIR, "all_DEGs.tsv")
TOP50_TSV = os.path.join(RESULTS_DIR, "top50_DEGs.tsv")
HEATMAP_PNG = os.path.join(RESULTS_DIR, "heatmap_sig.png")
PCA_PNG = os.path.join(RESULTS_DIR, "pca_plot.png")
TSNE_PNG = os.path.join(RESULTS_DIR, "tsne_plot.png")
UMAP_PNG = os.path.join(RESULTS_DIR, "umap_plot.png")

GROUP_POS = "Cancer"
GROUP_NEG = "Polyp"
PADJ_THRESH = 0.05
LFC_THRESH = 1.0
RANDOM_STATE = 0

# ----------------- FUNCTIONS -----------------
def safe_make_dir(path):
    os.makedirs(path, exist_ok=True)

def derive_group_from_title(title: str) -> str:
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"

def plot_density(gene_medians, out_png=None):
    plt.figure(figsize=(6,4))
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

def scatter_plot(X2, labels, title, xlabel, ylabel, outfile):
    plt.figure(figsize=(6,5))
    for g in np.unique(labels):
        sel = labels == g
        plt.scatter(X2[sel, 0], X2[sel, 1], label=g, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[OK] Saved {title} to {outfile}")

def volcano_plot(df, lfc_col="log2FC", padj_col="padj", lfc_thresh=1.0, padj_thresh=0.05, outfile=VOLCANO_PNG):
    df = df.dropna(subset=[lfc_col, padj_col]).copy()
    df["neglog10padj"] = -np.log10(df[padj_col].clip(lower=1e-300))
    sig = (df[padj_col] < padj_thresh) & (df[lfc_col].abs() >= lfc_thresh)
    up = sig & (df[lfc_col] > 0)
    dn = sig & (df[lfc_col] < 0)

    plt.figure(figsize=(7,6))
    plt.scatter(df.loc[~sig, lfc_col], df.loc[~sig, "neglog10padj"], s=10, alpha=0.6, label="NS")
    plt.scatter(df.loc[up, lfc_col], df.loc[up, "neglog10padj"], s=12, alpha=0.8, label="Signif up")
    plt.scatter(df.loc[dn, lfc_col], df.loc[dn, "neglog10padj"], s=12, alpha=0.8, label="Signif down")
    plt.axvline(+lfc_thresh, linestyle="--", linewidth=1)
    plt.axvline(-lfc_thresh, linestyle="--", linewidth=1)
    plt.axhline(-np.log10(padj_thresh), linestyle="--", linewidth=1)
    plt.xlabel(f"log2 fold change ({GROUP_POS} vs {GROUP_NEG})")
    plt.ylabel("-log10 adjusted p-value")
    plt.title("Volcano plot")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[OK] Saved volcano plot to {outfile}")

def plot_sig_heatmap(log_expr, res_df, meta, outfile, padj_thresh=PADJ_THRESH, lfc_thresh=LFC_THRESH):
    sig_genes = res_df.dropna(subset=["padj", "log2FC"])
    sig_genes = sig_genes[(sig_genes["padj"] < padj_thresh) & (sig_genes["log2FC"].abs() >= lfc_thresh)]
    if sig_genes.empty:
        print("[WARN] No significant DEGs found, skipping heatmap.")
        return
    expr_sig = log_expr.loc[sig_genes.index]
    expr_z = (expr_sig - expr_sig.mean(axis=1).values[:, None]) / expr_sig.std(axis=1).values[:, None]
    groups = meta.loc[expr_sig.columns, "group"]
    col_anno = HeatmapAnnotation(Group=groups.map({GROUP_POS:"red", GROUP_NEG:"blue"}), legend=True)
    cmp = ClusterMapPlotter(data=expr_z, col_cluster=True, row_cluster=True, top_annotation=col_anno, cmap="RdBu_r")
    plt.figure(figsize=(10,8))
    cmp.plot()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved DEG heatmap: {outfile}")

# ----------------- MAIN -----------------
def main():
    warnings.filterwarnings("ignore")
    safe_make_dir(RESULTS_DIR)

    # Load expression & metadata
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t", index_col=0)
    meta["group"] = meta["refinebio_title"].apply(derive_group_from_title)
    keep_samples = meta["group"].isin([GROUP_POS, GROUP_NEG])
    expr = expr.loc[:, keep_samples]
    meta = meta.loc[keep_samples]

    log_expr = np.log1p(expr)
    gene_medians = log_expr.median(axis=1)
    plot_density(gene_medians)

    # --- Wilcoxon rank sum test per gene ---
    results = []
    for gene in log_expr.index:
        group1 = log_expr.loc[gene, meta["group"]==GROUP_POS]
        group2 = log_expr.loc[gene, meta["group"]==GROUP_NEG]
        stat, p = ranksums(group1, group2)
        lfc = group1.mean() - group2.mean()
        results.append([gene, lfc, p])
    res_df = pd.DataFrame(results, columns=["gene","log2FC","pval"])
    res_df["padj"] = np.minimum(1, res_df["pval"]*len(res_df))  # Bonferroni correction
    res_df.set_index("gene", inplace=True)
    res_df.to_csv(ALL_TSV, sep="\t")
    res_df.head(50).to_csv(TOP50_TSV, sep="\t")
    print("[OK] Wilcoxon DE completed.")

    # --- Volcano plot ---
    volcano_plot(res_df)

    # --- Heatmap ---
    plot_sig_heatmap(log_expr, res_df, meta, HEATMAP_PNG)

    # --- PCA, t-SNE, UMAP ---
    X = log_expr.T.values
    labels = meta["group"].values
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pc = pca.fit_transform(X)
    scatter_plot(pc, labels, "PCA of samples", f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                 f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", PCA_PNG)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                random_state=RANDOM_STATE, perplexity=min(30, max(5,X.shape[0]//4)))
    ts = tsne.fit_transform(X)
    scatter_plot(ts, labels, "t-SNE of samples", "t-SNE1", "t-SNE2", TSNE_PNG)

    um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=RANDOM_STATE)
    uu = um.fit_transform(X)
    scatter_plot(uu, labels, "UMAP of samples", "UMAP1", "UMAP2", UMAP_PNG)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as smm
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


expr_df = pd.read_csv("./SRP068591.tsv", sep='\t', index_col=0)
meta_df = pd.read_csv("./metadata_SRP068591.tsv", sep='\t', index_col=0)
meta_df['group'] = meta_df['refinebio_disease']
expr_df = expr_df.loc[:, meta_df.index]

def top_variable_genes(expr_df, n):
    top_genes = expr_df.var(axis=1).nlargest(n).index
    return expr_df.loc[top_genes]

def run_hierarchical(expr_sub, k=3):
    data = expr_sub.T
    dist = pdist(data, metric="euclidean")
    link = linkage(dist, method="ward")
    clusters = fcluster(link, k, criterion="maxclust")
    return clusters

def chi_square_test(labels1, labels2):
    contingency = pd.crosstab(labels1, labels2)
    chi2, p, _, _ = chi2_contingency(contingency)
    return chi2, p


def plot_pca(expr_sub, clusters, n_genes):
    X = expr_sub.T
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    for cl in np.unique(clusters):
        sel = clusters == cl
        plt.scatter(pcs[sel, 0], pcs[sel, 1], label=f"Cluster {cl}", alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"Hierarchical Clustering PCA ({n_genes} genes)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"./results/hier_{n_genes}genes_pca.png", dpi=300)
    plt.close()
    print(f"[OK] Saved PCA plot → ./results/hier_{n_genes}genes_pca.png")


gene_counts = [10, 100, 1000, 5000, 10000]
k = 4

cluster_results = {}
for n in gene_counts:
    expr_sub = top_variable_genes(expr_df, n)
    clusters = run_hierarchical(expr_sub, k=k)
    meta_df[f"Cluster_{n}genes"] = clusters
    cluster_results[n] = clusters

    plot_pca(expr_sub, clusters, n)

chi_records = []

for i, n1 in enumerate(gene_counts):
    for n2 in gene_counts[i+1:]:
        chi2, p = chi_square_test(cluster_results[n1], cluster_results[n2])
        chi_records.append({
            "Comparison": f"{n1} vs {n2} genes",
            "Test": "Cluster vs Cluster",
            "Chi2": chi2,
            "p_value": p
        })

for n in gene_counts:
    cluster_series = pd.Series(cluster_results[n], index=meta_df.index)
    valid_idx = meta_df['group'].notna() & cluster_series.notna()
    if valid_idx.sum() == 0:
        print(f"Skipping {n} genes vs group: no valid data")
        continue
    chi2, p = chi_square_test(meta_df.loc[valid_idx, "group"], cluster_series[valid_idx])
    chi_records.append({
        "Comparison": f"{n} genes vs group",
        "Test": "Cluster vs Group",
        "Chi2": chi2,
        "p_value": p
    })

chi_df = pd.DataFrame(chi_records)
chi_df["adj_p_value"] = smm.multipletests(chi_df["p_value"], method="fdr_bh")[1]
chi_df = chi_df.sort_values("adj_p_value")
chi_df.to_csv("./results/assignment3_chi_squared_results.csv", index=False)

expr_5000 = top_variable_genes(expr_df, 5000)
expr_scaled = (expr_5000 - expr_5000.mean(axis=1).values[:, None]) / expr_5000.std(axis=1).values[:, None]

col_colors = pd.DataFrame({
    "Group": meta_df["group"],
    "Cluster (5000 genes)": meta_df["Cluster_5000genes"].astype(str)
})

group_palette = sns.color_palette("Set2", len(col_colors["Group"].unique()))
cluster_palette = sns.color_palette("tab10", len(col_colors["Cluster (5000 genes)"].unique()))
group_lut = dict(zip(col_colors["Group"].unique(), group_palette))
cluster_lut = dict(zip(col_colors["Cluster (5000 genes)"].unique(), cluster_palette))
col_colors_mapped = pd.DataFrame({
    "Group": col_colors["Group"].map(group_lut),
    "Cluster (5000 genes)": col_colors["Cluster (5000 genes)"].map(cluster_lut)
})

cg = sns.clustermap(
    expr_scaled,
    method="ward",
    metric="euclidean",
    col_colors=col_colors_mapped,
    cmap="vlag",
    figsize=(10, 10),
    xticklabels=False,
    yticklabels=False
)
plt.title("Hierarchical Clustering Heatmap (Top 5000 Genes)")
cg.savefig("./results/hierarchical_5000genes_heatmap.png", dpi=300)
plt.close()
print("[OK] Saved heatmap → ./results/hierarchical_5000genes_heatmap.png")

meta_df.to_csv("./results/assignment3_hierarchical_clusters_all.csv")

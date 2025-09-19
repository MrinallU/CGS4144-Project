# --- Setup ---
# pip install pandas numpy matplotlib scikit-learn umap-learn mygene
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import mygene

# -----------------------------
# Load data
# -----------------------------
expr = pd.read_csv("./SRP068591.tsv", sep="\t", index_col=0)
meta = pd.read_csv("./metadata_SRP068591.tsv", sep="\t")

# Make sure metadata is indexed by sample ID and aligned to expression columns
if "refinebio_accession_code" not in meta.columns:
    raise KeyError("Expected 'refinebio_accession_code' column in metadata.")

meta = meta.set_index("refinebio_accession_code")
expr = expr.loc[:, expr.columns.intersection(meta.index)]
meta = meta.loc[expr.columns]

print("Matrix size (genes x samples):", expr.shape)
print("# unique genes (row index):", expr.index.nunique())

# -----------------------------
# Ensembl → HGNC symbols (safe)
# -----------------------------
ens = pd.Index(expr.index).str.replace(r"\.\d+$", "", regex=True)

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
unmapped_n = (~mapped_mask).sum()
dup_symbols = pd.Series(new_index[mapped_mask]).value_counts()
dup_n = (dup_symbols > 1).sum()

print(f"Genes total: {len(ens)}")
print(f"Mapped to symbols: {mapped_mask.sum()}")
print(f"Unmapped: {unmapped_n}")
print(f"Symbols with duplicates (to be collapsed): {dup_n}")

expr_mapped = expr.loc[mapped_mask].copy()
expr_mapped.index = new_index[mapped_mask]
expr_mapped = expr_mapped.groupby(level=0, sort=False).median()

print("After mapping/collapsing -> shape:", expr_mapped.shape)

# -----------------------------
# Log transform + median density
# -----------------------------
log_expr = np.log1p(expr_mapped)
gene_medians = log_expr.median(axis=1)

plt.figure(figsize=(6, 4))
gene_medians.plot(kind="density")
plt.title("Per-gene median (log1p counts) density")
plt.xlabel("Median log1p expression per gene")
plt.ylabel("Density")
plt.tight_layout()
plt.show()


# -----------------------------
# Build grouping for coloring
# Priority: refinebio_disease; fallback: derive from refinebio_title
# -----------------------------
def derive_group_from_title(title: str) -> str:
    """
    Map SRP068591 titles to 2 groups:
      - Polyp: SSA/P, HP, AP
      - Cancer: CA, CR, CL, UR, UL
    Everything else -> Other
    """
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"


group_col = None
if "refinebio_disease" in meta.columns and meta["refinebio_disease"].notna().any():
    # Use as-is if it contains 2+ categories
    if meta["refinebio_disease"].nunique() >= 2:
        group_col = "refinebio_disease"

if group_col is None:
    # Derive from title
    if "refinebio_title" not in meta.columns:
        raise KeyError(
            "Metadata lacks both 'refinebio_disease' and 'refinebio_title' to create groups."
        )
    meta["group"] = meta["refinebio_title"].apply(derive_group_from_title)
    # Keep only Polyp vs Cancer, drop 'Other'
    keep = meta["group"].isin(["Polyp", "Cancer"])
    meta = meta.loc[keep]
    log_expr = log_expr.loc[:, meta.index]  # realign matrix to kept samples
    group_col = "group"

# Sanity checks
if meta[group_col].nunique() < 2:
    # Show a quick preview to help debug
    print("Unique values in chosen group column:", meta[group_col].unique())
    raise ValueError(
        f"Grouping column '{group_col}' has fewer than 2 levels after filtering."
    )

print(f"Using grouping column: {group_col}")
print(meta[group_col].value_counts())

# -----------------------------
# Matrix for embeddings (center genes)
# -----------------------------
X = (log_expr.T - log_expr.T.mean()).values  # samples x genes
groups = meta.loc[log_expr.columns, group_col].astype(str).values
labels = np.array(groups)
unique_groups = np.unique(labels)

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2, random_state=0)
pc = pca.fit_transform(X)

plt.figure(figsize=(6, 5))
for g in unique_groups:
    sel = labels == g
    plt.scatter(pc[sel, 0], pc[sel, 1], label=g, alpha=0.85)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
plt.title("PCA of samples")
plt.legend(frameon=False, title=group_col)
plt.tight_layout()
plt.show()

# -----------------------------
# t-SNE
# -----------------------------
tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    random_state=0,
    perplexity=min(30, max(5, X.shape[0] // 4)),
)
ts = tsne.fit_transform(X)

plt.figure(figsize=(6, 5))
for g in unique_groups:
    sel = labels == g
    plt.scatter(ts[sel, 0], ts[sel, 1], label=g, alpha=0.85)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of samples")
plt.legend(frameon=False, title=group_col)
plt.tight_layout()
plt.show()

# -----------------------------
# UMAP
# -----------------------------
um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
uu = um.fit_transform(X)

plt.figure(figsize=(6, 5))
for g in unique_groups:
    sel = labels == g
    plt.scatter(uu[sel, 0], uu[sel, 1], label=g, alpha=0.85)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of samples")
plt.legend(frameon=False, title=group_col)
plt.tight_layout()
plt.show()

# -----------------------------
# Brief textual summary printed to console
# -----------------------------
print("\n===== SUMMARY =====")
print(
    f"Expression matrix (after mapping): {expr_mapped.shape[0]} genes x {log_expr.shape[1]} samples"
)
print("Per-gene median (log1p) summary:")
print(gene_medians.describe()[["min", "25%", "50%", "75%", "max"]])
print(f"Groups used: {dict(meta[group_col].value_counts())}")
print(
    f"PCA variance explained: PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, "
    f"PC2={pca.explained_variance_ratio_[1] * 100:.1f}%"
)

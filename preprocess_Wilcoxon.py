#!/usr/bin/env python3
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
RESULTS_DIR = "./results"

GROUP_POS = "Cancer"
GROUP_NEG = "Polyp"


def safe_make_dir(path):
    os.makedirs(path, exist_ok=True)


def derive_group_from_title(title: str) -> str:
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"


def plot_density(gene_medians: pd.Series, out_png: str):
    plt.figure(figsize=(6, 4))
    gene_medians.plot(kind="density")
    plt.title("Per-gene median (log1p counts) density")
    plt.xlabel("Median log1p expression per gene")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[OK] Saved density plot: {out_png}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    safe_make_dir(RESULTS_DIR)

    # Load data
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t")
    meta = meta.set_index("refinebio_accession_code")
    common = expr.columns.intersection(meta.index)
    expr = expr.loc[:, common]
    meta = meta.loc[common]

    print(f"Matrix size (genes x samples): {expr.shape}")

    # log transform
    log_expr = np.log1p(expr)

    # Assign groups
    meta["group"] = meta["refinebio_title"].apply(derive_group_from_title)
    meta = meta.loc[meta["group"].isin([GROUP_NEG, GROUP_POS])]
    log_expr = log_expr.loc[:, meta.index]

    print("\n[Groups]")
    print(meta["group"].value_counts())

    # Density plot of per-gene medians
    gene_medians = log_expr.median(axis=1)
    plot_density(gene_medians, os.path.join(RESULTS_DIR, "density.png"))

    # Save processed data
    log_expr.to_csv(os.path.join(RESULTS_DIR, "log_expr.tsv"), sep="\t")
    meta.to_csv(os.path.join(RESULTS_DIR, "meta_processed.tsv"), sep="\t")
    print(f"[OK] Saved log_expr.tsv and meta_processed.tsv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

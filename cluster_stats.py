#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ================================
# Config
# ================================
RESULTS_DIR = "./results"
META_PATH = "./metadata_SRP068591.tsv"
KMEANS_LABELS = os.path.join(RESULTS_DIR, "kmeans_labels.csv")
OUT_TSV = os.path.join(RESULTS_DIR, "chi2_cluster_vs_group.tsv")

# If you later add other clustering results, list them here:
CLUSTER_METHODS = {
    "KMeans": KMEANS_LABELS,
    # "Hierarchical": "./results/hier_labels.csv",
    # "Spectral": "./results/spectral_labels.csv",
}


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


def run_chi2_test(labels: pd.Series, groups: pd.Series):
    """Run chi-squared test of independence between clusters and biological groups."""
    contingency = pd.crosstab(labels, groups)
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        "Chi2": chi2,
        "df": dof,
        "p_value": p,
        "n_clusters": len(labels.unique()),
        "n_groups": len(groups.unique()),
    }


# ================================
# Main
# ================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(META_PATH, sep="\t")
    meta = meta.set_index("refinebio_accession_code")
    meta["group"] = meta["refinebio_title"].apply(derive_group)

    results = []
    for name, path in CLUSTER_METHODS.items():
        if not os.path.exists(path):
            print(f"[WARN] Skipping {name}: missing label file {path}")
            continue

        labels = pd.read_csv(path, index_col=0).iloc[:, 0]
        labels.name = "cluster"

        # Align with metadata
        common = meta.index.intersection(labels.index)
        labels = labels.loc[common]
        groups = meta.loc[common, "group"]

        # Skip if insufficient variety
        if groups.nunique() < 2 or labels.nunique() < 2:
            print(f"[WARN] {name}: insufficient group/cluster diversity, skipping.")
            continue

        print(f"[INFO] Running chi2 test for {name} ({len(common)} samples)...")
        stats = run_chi2_test(labels, groups)
        stats["Method"] = name
        results.append(stats)

    # Combine into DataFrame
    if not results:
        raise RuntimeError("No clustering results found or tests failed.")

    df = pd.DataFrame(results)

    # Adjust for multiple testing
    df["p_adj_bonferroni"] = multipletests(df["p_value"], method="bonferroni")[1]
    df["p_adj_fdr_bh"] = multipletests(df["p_value"], method="fdr_bh")[1]

    # Save
    df = df[
        [
            "Method",
            "Chi2",
            "df",
            "p_value",
            "p_adj_bonferroni",
            "p_adj_fdr_bh",
            "n_clusters",
            "n_groups",
        ]
    ].sort_values("p_value")
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[OK] Saved chi-squared results → {OUT_TSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

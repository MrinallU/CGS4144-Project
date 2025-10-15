#!/usr/bin/env python3
import os
import pandas as pd
from gprofiler import GProfiler

RESULTS_DIR = "./results"
PROC_EXPR = os.path.join(RESULTS_DIR, "log_expr.tsv")
PROC_META = os.path.join(RESULTS_DIR, "meta.tsv")
DIFF_EXPR = os.path.join(RESULTS_DIR, "all_wilcoxon.tsv")
ENRICH_TABLE = os.path.join(RESULTS_DIR, "enrichment_results.tsv")


def main():
    # Load Wilcoxon results
    de = pd.read_csv(DIFF_EXPR, sep="\t")
    print("Columns in DE file:", de.columns.tolist())
    print(de.head())

    # Select significantly DE genes (e.g., adj p < 0.05)
    sig = de[de["pval"] < 0.05].sort_values("pval"
                                            "")
    sig_genes = sig["gene"].tolist()

    if not sig_genes:
        print("[WARN] No significant genes found at adj p < 0.05")
        return

    # Run enrichment with gProfiler
    gp = GProfiler(return_dataframe=True)
    enr = gp.profile(
        organism="hsapiens",
        query=sig_genes,
        sources=["GO:BP"],  # Gene Ontology: Biological Process
    )

    # Save enrichment results
    enr.to_csv(ENRICH_TABLE, sep="\t", index=False)
    print(f"[OK] Saved enrichment results to {ENRICH_TABLE}")


if __name__ == "__main__":
    main()

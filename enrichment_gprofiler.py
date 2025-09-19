#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run gene set enrichment analysis on DE results and optionally merge team results.

Method: g:Profiler (Python client)
Default ontology: GO:BP (biological process)

Usage examples
--------------
# 1) Run enrichment on your DE table with defaults (GO:BP; padj<0.05, |log2FC|>=1)
python enrichment_gprofiler.py

# 2) Custom thresholds and ontology (e.g., GO:MF)
python enrichment_gprofiler.py --deg-tsv ./results/all_DEGs.tsv \
    --padj 0.05 --lfc 1.0 --ontology GO:MF

# 3) Merge team results (pass any number of enrichment TSVs from different methods)
python enrichment_gprofiler.py --merge ./results/gprof_all.tsv ./results/topGO_all.tsv \
    --out-merged ./results/team_joint_enrichment.tsv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

try:
    from gprofiler import GProfiler
except ImportError:
    raise SystemExit(
        "Missing dependency: gprofiler-official\n"
        "Install with: pip install gprofiler-official"
    )

RESULTS_DIR = "./results"

# Default input (from your DE pipeline)
DEFAULT_DEG_TSV = os.path.join(RESULTS_DIR, "all_DEGs.tsv")

# Default output files
OUT_ALL = os.path.join(RESULTS_DIR, "gprof_all.tsv")
OUT_UP = os.path.join(RESULTS_DIR, "gprof_up.tsv")
OUT_DN = os.path.join(RESULTS_DIR, "gprof_down.tsv")

# g:Profiler source keys you can choose for `--ontology`
#   GO:BP, GO:MF, GO:CC, REAC (Reactome), WP (WikiPathways),
#   HP (Human Phenotype), CORUM, HPA, TF, MIRNA
VALID_SOURCES = {
    "GO:BP",
    "GO:MF",
    "GO:CC",
    "REAC",
    "WP",
    "HP",
    "CORUM",
    "HPA",
    "TF",
    "MIRNA",
}


def load_deg_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DE table not found: {path}")
    df = pd.read_csv(path, sep="\t", header=0, index_col=0)
    required = {"log2FoldChange", "padj"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"DE table must contain columns {required}, got: {df.columns.tolist()}"
        )
    return df


def pick_sig_genes(
    df: pd.DataFrame, padj: float, lfc: float
) -> Tuple[List[str], List[str], List[str]]:
    use = df.dropna(subset=["padj", "log2FoldChange"]).copy()
    sig = (use["padj"] < padj) & (use["log2FoldChange"].abs() >= lfc)
    sig_df = use.loc[sig]
    up_df = sig_df.loc[sig_df["log2FoldChange"] > 0]
    dn_df = sig_df.loc[sig_df["log2FoldChange"] < 0]
    return sig_df.index.tolist(), up_df.index.tolist(), dn_df.index.tolist()


def run_gprofiler(
    gene_list: List[str], source: str, out_tsv: str, organism: str = "hsapiens"
) -> Optional[pd.DataFrame]:
    if not gene_list:
        print(f"[WARN] No genes supplied for enrichment ({out_tsv}); skipping.")
        return None

    gp = GProfiler(return_dataframe=True)
    # Keep it simple: one source at a time
    print(f"[INFO] Running g:Profiler on {len(gene_list)} genes | source={source}")
    res = gp.profile(
        organism=organism,
        query=gene_list,
        sources=[source],
        user_threshold=1.0,  # let gProfiler compute its FDR; we’ll filter later if needed
        no_iea=False,  # include electronic annotations
        significance_threshold_method="g_SCS",  # g:Profiler default multiple testing
    )

    if res is None or res.empty:
        print(f"[INFO] g:Profiler returned no results for {out_tsv}")
        return None

    # Normalize/rename a few columns for clarity
    keep_cols = [
        "source",
        "native",
        "name",
        "p_value",
        "term_size",
        "query_size",
        "intersection_size",
        "precision",
        "recall",
        "effective_domain_size",
        "source_order",
        "intersections",
    ]
    # Keep only those that exist
    keep_cols = [c for c in keep_cols if c in res.columns]
    res = res.loc[:, keep_cols].copy()
    # For convenience, add -log10 p
    res["minus_log10_p"] = -np.log10(res["p_value"].clip(lower=1e-300))

    # Save
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    res.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Saved enrichment: {os.path.abspath(out_tsv)}")
    return res


def merge_team_results(paths: List[str], outfile: str) -> pd.DataFrame:
    """
    Merge multiple enrichment TSVs (from different methods) into a joint table.
    Expected minimal columns in each file: source, native (term ID), name, p_value
    Additional method-specific columns are preserved with suffixes.

    We produce:
      - one row per unique (source, term_id/native)
      - columns: name (first seen), per-method p/q/other (if available),
                 methods_significant (count of p<0.05 across inputs),
                 methods_tested (how many methods included this term at all)
    """
    if not paths:
        raise ValueError("No input files provided to --merge")

    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] merge: file not found, skipping: {p}")
            continue
        df = pd.read_csv(p, sep="\t", header=0)
        # Try to infer method name from filename (e.g., gprof, topGO, clusterProfiler)
        method = os.path.splitext(os.path.basename(p))[0]
        method = method.replace(".tsv", "").replace(".txt", "")
        # Normalize essential columns
        rename = {}
        if "native" not in df.columns:
            # Some tools might call it 'term_id' or 'id'
            cand = [c for c in ["term_id", "id"] if c in df.columns]
            if cand:
                rename[cand[0]] = "native"
        if "name" not in df.columns:
            cand = [c for c in ["term_name", "description"] if c in df.columns]
            if cand:
                rename[cand[0]] = "name"
        if rename:
            df = df.rename(columns=rename)
        must = {"source", "native", "name"}
        if not must.issubset(df.columns):
            print(f"[WARN] merge: {p} missing required cols {must}; skipping")
            continue

        # Keep only minimal + common stats
        keep = ["source", "native", "name"]
        for stat in ["p_value", "adjusted_p_value", "q_value", "minus_log10_p"]:
            if stat in df.columns:
                keep.append(stat)
        df = df.loc[:, keep].copy()
        # Add method prefix to stat columns
        stat_cols = [c for c in df.columns if c not in ["source", "native", "name"]]
        df = df.rename(columns={c: f"{method}__{c}" for c in stat_cols})
        frames.append(df)

    if not frames:
        raise ValueError("No valid enrichment files to merge.")

    # Outer join across all methods on (source, native)
    joint = frames[0]
    for f in frames[1:]:
        joint = joint.merge(f, on=["source", "native", "name"], how="outer")

    # Compute method counts
    # methods_tested: any method with any stat present for the term
    method_names = set()
    for c in joint.columns:
        if "__" in c:
            method_names.add(c.split("__", 1)[0])

    def count_tested(row):
        n = 0
        for m in method_names:
            has_any = any(
                pd.notna(row[f"{m}__{stat}"])
                for stat in ["p_value", "adjusted_p_value", "q_value", "minus_log10_p"]
                if f"{m}__{stat}" in joint.columns
            )
            n += int(has_any)
        return n

    def count_sig(row, alpha=0.05):
        n = 0
        for m in method_names:
            pv = row.get(f"{m}__p_value", np.nan)
            qv = row.get(f"{m}__adjusted_p_value", np.nan)
            # Prefer q-value / adjusted p if present
            val = np.nanmin([qv, pv]) if not (np.isnan(qv) and np.isnan(pv)) else np.nan
            if pd.notna(val) and val < alpha:
                n += 1
        return n

    joint["methods_tested"] = joint.apply(count_tested, axis=1)
    joint["methods_significant"] = joint.apply(count_sig, axis=1)

    # Sort by how often significant, then alphabetically
    joint = joint.sort_values(
        ["methods_significant", "source", "native"], ascending=[False, True, True]
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    joint.to_csv(outfile, sep="\t", index=False)
    print(f"[OK] Saved team joint table: {os.path.abspath(outfile)}")
    return joint


def main():
    ap = argparse.ArgumentParser(
        description="Run g:Profiler enrichment and optionally merge team results."
    )
    ap.add_argument(
        "--deg-tsv",
        default=DEFAULT_DEG_TSV,
        help="Path to DE results (TSV with log2FoldChange & padj).",
    )
    ap.add_argument(
        "--padj", type=float, default=0.05, help="Adjusted p-value threshold for DEGs."
    )
    ap.add_argument(
        "--lfc",
        type=float,
        default=1.0,
        help="Absolute log2 fold-change threshold for DEGs.",
    )
    ap.add_argument(
        "--ontology",
        default="GO:BP",
        choices=sorted(list(VALID_SOURCES)),
        help="Ontology/source to use for enrichment.",
    )
    ap.add_argument(
        "--organism",
        default="hsapiens",
        help="Organism code for g:Profiler (default: hsapiens).",
    )
    ap.add_argument(
        "--out-all",
        default=OUT_ALL,
        help="Output TSV for all significant DEGs enrichment.",
    )
    ap.add_argument(
        "--out-up", default=OUT_UP, help="Output TSV for up-regulated DEGs enrichment."
    )
    ap.add_argument(
        "--out-down",
        default=OUT_DN,
        help="Output TSV for down-regulated DEGs enrichment.",
    )
    ap.add_argument(
        "--merge",
        nargs="*",
        help="Optional: list of enrichment TSVs (from different methods) to merge.",
    )
    ap.add_argument(
        "--out-merged",
        default=os.path.join(RESULTS_DIR, "team_joint_enrichment.tsv"),
        help="Output TSV for the merged team table.",
    )
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load DE and pick significant genes
    deg = load_deg_table(args.deg_tsv)
    sig, up, dn = pick_sig_genes(deg, padj=args.padj, lfc=args.lfc)
    print(f"[INFO] Significant DEGs: total={len(sig)} | up={len(up)} | down={len(dn)}")

    # 2) Run enrichment (our method = gProfiler; our ontology = args.ontology)
    run_gprofiler(sig, args.ontology, args.out_all, organism=args.organism)
    run_gprofiler(up, args.ontology, args.out_up, organism=args.organism)
    run_gprofiler(dn, args.ontology, args.out_down, organism=args.organism)

    # 3) Optional: merge multiple methods’ enrichment results into one joint table
    if args.merge:
        merge_team_results(args.merge, args.out_merged)
    else:
        print(
            "[INFO] Skipping team merge. Provide TSVs via --merge to build the joint table."
        )


if __name__ == "__main__":
    main()

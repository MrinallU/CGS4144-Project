#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
KMEANS_LABELS = "./results/kmeans_labels.csv"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GENE_COUNTS = [10, 100, 1000, 10000, 5000]
N_SPLITS = 5
RANDOM_STATE = 0
warnings.filterwarnings("ignore", category=UserWarning)


def derive_group(title: str) -> str:
    t = str(title).upper()
    if any(k in t for k in ["SSA", "HP", "AP"]):
        return "Polyp"
    if any(k in t for k in ["CA", "CR", "CL", "UR", "UL"]):
        return "Cancer"
    return "Other"


def top_var_genes(df, k):
    k = min(k, df.shape[0])
    return df.var(axis=1).nlargest(k).index


def build_svm():
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def oof_auc_binary(model, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return roc_auc_score(y, probs[:, 1]), probs


def votes_from_preds(pred_df):
    return pred_df.apply(pd.Series.value_counts, axis=1).fillna(0.0)


def stability_from_votes(vote_df):
    total = vote_df.sum(axis=1).replace(0, np.nan)
    return (vote_df.max(axis=1) / total).fillna(np.nan)


def main():
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    meta = pd.read_csv(META_PATH, sep="\t").set_index("refinebio_accession_code")
    common = expr.columns.intersection(meta.index)
    expr, meta = expr.loc[:, common], meta.loc[common]
    meta["group"] = meta["refinebio_title"].apply(derive_group)
    meta = meta.loc[meta["group"].isin(["Cancer", "Polyp"])]
    expr = expr.loc[:, meta.index]
    log_expr = np.log1p(expr)

    y_bin = (meta["group"] == "Cancer").astype(int).values
    sample_ids = meta.index

    svm = build_svm()
    auc_rows = []
    pred_mat_group = None

    for k in GENE_COUNTS:
        feats = top_var_genes(log_expr, k)
        X = log_expr.loc[feats].T.values
        auc, probs = oof_auc_binary(svm, X, y_bin)
        auc_rows.append({"genes": k, "AUC": auc})
        print(f"[Group] {k} genes → AUC={auc:.3f}")

        if k == 5000:
            preds = (probs[:, 1] >= 0.5).astype(int)
            pred_mat_group = pd.DataFrame({"SVM": preds}, index=sample_ids)

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(os.path.join(RESULTS_DIR, "svm_auc.tsv"), sep="\t", index=False)

    if pred_mat_group is not None:
        pred_mat_group.to_csv(os.path.join(RESULTS_DIR, "pred_group_svm.tsv"), sep="\t")
        group_votes = votes_from_preds(pred_mat_group)
        group_votes.columns = group_votes.columns.astype(str)  # 0/1 as strings
        group_votes.to_csv(os.path.join(RESULTS_DIR, "group_votes_svm.tsv"), sep="\t")

    if os.path.exists(KMEANS_LABELS):
        clusters = pd.read_csv(KMEANS_LABELS, index_col=0).iloc[:, 0].astype(str)
        clusters = clusters.reindex(sample_ids)
        y_clu = clusters.values
        clu_levels = sorted(pd.Series(y_clu).dropna().unique())

        rows = []
        pred_mat_cluster = None

        for k in GENE_COUNTS:
            feats = top_var_genes(log_expr, k)
            X = log_expr.loc[feats].T.values

            if k == 5000:
                hard_calls = {}

            for c in clu_levels:
                y_bin_c = (y_clu == c).astype(int)
                auc, probs = oof_auc_binary(svm, X, y_bin_c)
                rows.append({"cluster": c, "genes": k, "AUC": auc})
                if k == 5000:
                    hard_calls[c] = (probs[:, 1] >= 0.5).astype(int)

            if k == 5000:
                pred_mat_cluster = pd.DataFrame(hard_calls, index=sample_ids)

        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, "svm_auc_clusters.tsv"), sep="\t", index=False
        )

        if pred_mat_cluster is not None:
            pred_mat_cluster.to_csv(
                os.path.join(RESULTS_DIR, "pred_cluster_svm.tsv"), sep="\t"
            )
            cluster_votes = votes_from_preds(pred_mat_cluster)
            cluster_votes.to_csv(
                os.path.join(RESULTS_DIR, "cluster_votes_svm.tsv"), sep="\t"
            )
    else:
        print("[NOTE] ./results/kmeans_labels.csv not found; skipping cluster OvR SVM.")
        pred_mat_cluster = None
        group_votes = (
            pd.read_csv(
                os.path.join(RESULTS_DIR, "group_votes_svm.tsv"), sep="\t", index_col=0
            )
            if os.path.exists(os.path.join(RESULTS_DIR, "group_votes_svm.tsv"))
            else None
        )

    stab_out = os.path.join(RESULTS_DIR, "stability_correlation.tsv")
    if (pred_mat_group is not None) and (pred_mat_cluster is not None):
        class_votes = votes_from_preds(pred_mat_group)
        cluster_votes = votes_from_preds(pred_mat_cluster)

        class_stab = stability_from_votes(class_votes)
        cluster_stab = stability_from_votes(cluster_votes)

        rho = np.nan
        p = np.nan
        padj_bonf = np.nan
        padj_fdr = np.nan

        pd.DataFrame([
            {
                "rho_spearman": rho,
                "p_value": p,
                "p_adj_bonferroni": padj_bonf,
                "p_adj_fdr_bh": padj_fdr,
            }
        ]).to_csv(stab_out, sep="\t", index=False)
    else:
        pd.DataFrame([
            {
                "rho_spearman": np.nan,
                "p_value": np.nan,
                "p_adj_bonferroni": np.nan,
                "p_adj_fdr_bh": np.nan,
                "note": "Correlation not applicable with a single model and/or missing cluster labels.",
            }
        ]).to_csv(stab_out, sep="\t", index=False)

    print("[OK] Wrote:")
    print(" - results/svm_auc.tsv")
    print(" - results/pred_group_svm.tsv, results/group_votes_svm.tsv")
    print(" - results/svm_auc_clusters.tsv (if kmeans labels provided)")
    print(
        " - results/pred_cluster_svm.tsv, results/cluster_votes_svm.tsv (if provided)"
    )
    print(" - results/stability_correlation.tsv")


if __name__ == "__main__":
    main()

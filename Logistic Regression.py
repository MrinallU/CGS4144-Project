#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
EXPR_PATH = "./SRP068591.tsv"
META_PATH = "./metadata_SRP068591.tsv"
CLUSTERS_PATH = "./results/assignment3_hierarchical_clusters_all.csv"
RESULTS_DIR = "./results"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

GENE_COUNTS = [10, 100, 1000, 5000, 10000]
RANDOM_STATE = 0
CV_FOLDS = 5
TOP_FEATURES_PER_CLASS = 30  # how many top coef genes to include per OvR model

# ---------- LOAD DATA ----------
expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
meta = pd.read_csv(META_PATH, sep="\t", index_col=0)
meta["group"] = meta["refinebio_disease"]
clusters_df = pd.read_csv(CLUSTERS_PATH, index_col=0)

# align samples
common = expr.columns.intersection(meta.index).intersection(clusters_df.index)
expr = expr[common]
meta = meta.loc[common].copy()
clusters_df = clusters_df.loc[common].copy()
meta["cluster"] = clusters_df["Cluster_5000genes"].astype(str)

# log transform
log_expr = np.log1p(expr)

# helper
def get_top_genes(expr_df, n):
    vars = expr_df.var(axis=1)
    return vars.sort_values(ascending=False).head(n).index

# containers
auc_records = []
all_signature_genes = set()
prediction_files = []  # track generated prediction files

# ---------- MAIN: loop gene counts ----------
for n in GENE_COUNTS:
    print(f"\n=== Running models for top {n} genes ===")
    top_genes = get_top_genes(log_expr, n)
    expr_sub = log_expr.loc[top_genes]
    # Build X as DataFrame with sample index preserved
    X_df = pd.DataFrame(StandardScaler().fit_transform(expr_sub.T),
                        index=expr_sub.columns,
                        columns=expr_sub.index)

    # ---------- (1) Binary: Cancer vs Polyp ----------
    y_group = meta["group"].replace({"Cancer": 1, "Polyp": 0})
    Xg, yg = X_df.align(y_group, join="inner", axis=0)
    yg_mask = yg.notna()
    Xg = Xg.loc[yg_mask]
    yg = yg.loc[yg_mask]
    if len(yg.unique()) > 1 and len(yg) >= CV_FOLDS:
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # out-of-fold probabilities
        probs = cross_val_predict(clf, Xg, yg, cv=skf, method="predict_proba")
        auc = roc_auc_score(yg, probs[:,1])
        print(f"[AUC] Cancer vs Polyp ({n} genes): {auc:.3f}")
        auc_records.append({"Target":"Cancer_vs_Polyp","Genes":n,"AUC":auc})
        # fit on full data to extract coefficients and predictions (for signature & sample matrix)
        clf.fit(Xg, yg)
        preds = clf.predict(Xg)
        df_preds = pd.DataFrame({
            "sample": Xg.index,
            "pred_label": preds.astype(str),
            "pred_prob_class1": probs[:,1]
        }).set_index("sample")
        outpath = os.path.join(PRED_DIR, f"logreg_group_{n}genes_preds.tsv")
        df_preds.to_csv(outpath, sep="\t")
        prediction_files.append(outpath)
        # extract top features by absolute coef
        coefs = pd.Series(np.abs(clf.coef_.ravel()), index=Xg.columns)
        top_feats = coefs.nlargest(TOP_FEATURES_PER_CLASS).index.tolist()
        all_signature_genes.update(top_feats)
    else:
        print(f"[WARN] Not enough samples/labels for group model at {n} genes; skipping.")

    # ---------- (2) Multiclass: Hierarchical clusters (OvR) ----------
    y_cluster = meta["cluster"].astype(str)
    Xc, yc = X_df.align(y_cluster, join="inner", axis=0)
    yc_mask = yc.notna()
    Xc = Xc.loc[yc_mask]
    yc = yc.loc[yc_mask]
    if len(yc.unique()) > 1 and len(yc) >= CV_FOLDS:
        clf = LogisticRegression(max_iter=2000, solver="liblinear", multi_class="ovr")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        probs = cross_val_predict(clf, Xc, yc, cv=skf, method="predict_proba")
        # multiclass AUC (OvR)
        y_bin = label_binarize(yc, classes=np.unique(yc))
        auc = roc_auc_score(y_bin, probs, multi_class="ovr")
        print(f"[AUC] Hierarchical clusters ({n} genes): {auc:.3f}")
        auc_records.append({"Target":"Hierarchical_Clusters","Genes":n,"AUC":auc})
        # fit final model and save preds
        clf.fit(Xc, yc)
        preds = clf.predict(Xc)
        df_preds = pd.DataFrame({
            "sample": Xc.index,
            "pred_label": preds.astype(str)
        }).set_index("sample")
        # also add predicted probability for predicted class (max prob)
        df_preds["pred_prob_max"] = probs.max(axis=1)
        outpath = os.path.join(PRED_DIR, f"logreg_cluster_{n}genes_preds.tsv")
        df_preds.to_csv(outpath, sep="\t")
        prediction_files.append(outpath)
        # extract top features per OvR class
        # clf.coef_ shape: (n_classes, n_features)
        classes = clf.classes_
        coefs = pd.DataFrame(clf.coef_, index=classes, columns=Xc.columns)
        for cls in classes:
            top_feats = coefs.loc[cls].abs().nlargest(TOP_FEATURES_PER_CLASS).index.tolist()
            all_signature_genes.update(top_feats)
    else:
        print(f"[WARN] Not enough samples/labels for cluster model at {n} genes; skipping.")

# ---------- SAVE AUC SUMMARY ----------
auc_df = pd.DataFrame(auc_records)
auc_df.to_csv(os.path.join(RESULTS_DIR, "logreg_auc_summary.tsv"), sep="\t", index=False)
print(f"\n[OK] Saved AUC summary → {os.path.join(RESULTS_DIR, 'logreg_auc_summary.tsv')}")

# ---------- Build union signature heatmap ----------
if len(all_signature_genes) == 0:
    print("[WARN] No signature genes collected; skipping signature heatmap.")
else:
    sig_genes = sorted(list(all_signature_genes))
    print(f"[INFO] Number of genes in union signature: {len(sig_genes)}")
    expr_sig = log_expr.loc[sig_genes, :]  # rows=genes, cols=samples
    # z-score by gene
    zexpr = (expr_sig.sub(expr_sig.mean(axis=1), axis=0)).div(expr_sig.std(axis=1), axis=0)

    # build annotation DataFrame (Assignment1 group)
    ann = pd.DataFrame({
        "Group": meta["group"],
        "Cluster_5000": meta["cluster"]
    }, index=meta.index).loc[expr_sig.columns]

    # map colors
    group_cats = ann["Group"].unique().tolist()
    group_palette = dict(zip(group_cats, sns.color_palette("Set2", len(group_cats))))
    cluster_cats = sorted(ann["Cluster_5000"].unique().tolist())
    cluster_palette = dict(zip(cluster_cats, sns.color_palette("tab10", len(cluster_cats))))
    col_colors = pd.DataFrame({
        "Group": ann["Group"].map(group_palette),
        "Cluster": ann["Cluster_5000"].map(cluster_palette)
    }, index=ann.index)

    # plot clustermap
    cg = sns.clustermap(
        zexpr,
        row_cluster=True,
        col_cluster=True,
        cmap="RdBu_r",
        col_colors=col_colors,
        figsize=(12, 10),
        xticklabels=False,
        yticklabels=True
    )
    # add legends
    for label, color in group_palette.items():
        cg.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    for label, color in cluster_palette.items():
        cg.ax_col_dendrogram.bar(0, 0, color=color, label=f"Cluster {label}", linewidth=0)
    cg.ax_col_dendrogram.legend(loc="center", ncol=3, bbox_to_anchor=(0.5, 1.2), frameon=False)
    out_heat = os.path.join(RESULTS_DIR, "logreg_union_signature_heatmap.png")
    cg.savefig(out_heat, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved signature heatmap → {out_heat}")

# ---------- Build sample × model matrix by reading all prediction files ----------
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.tsv")))
if len(pred_files) == 0:
    print("[WARN] No prediction files found in ./results/predictions/ to merge.")
else:
    preds_list = []
    for pf in pred_files:
        df = pd.read_csv(pf, sep="\t", index_col=0)
        # Create a simple label column name from filename
        name = os.path.splitext(os.path.basename(pf))[0]
        # add label column (use pred_label if present or predict class from prob)
        if "pred_label" in df.columns:
            col = df["pred_label"].astype(str)
        elif "pred_prob_class1" in df.columns:
            col = (df["pred_prob_class1"] >= 0.5).astype(int).astype(str)
        else:
            # fallback: first column
            col = df.iloc[:,0].astype(str)
        col.name = name
        preds_list.append(col)
    pred_matrix = pd.concat(preds_list, axis=1).sort_index()
    pred_matrix.to_csv(os.path.join(RESULTS_DIR, "sample_by_model_matrix.tsv"), sep="\t")
    print(f"[OK] Saved sample × model matrix → {os.path.join(RESULTS_DIR, 'sample_by_model_matrix.tsv')}")

    # ---------- compute per-sample consensus stats ----------
    # (a) for each sample, count how many models predict each class label (we store as dict)
    class_counts = pred_matrix.apply(lambda row: row.value_counts().to_dict(), axis=1)
    # For convenience, compute per-sample top-class count (how many models agree on modal class)
    top_counts = pred_matrix.apply(lambda row: row.value_counts().max(), axis=1)
    # (b) how many models predict the same cluster? For files that are cluster models include those columns
    # define cluster columns as those with 'cluster' in filename
    cluster_cols = [c for c in pred_matrix.columns if "cluster" in c]
    if len(cluster_cols) > 0:
        cluster_mode_counts = pred_matrix[cluster_cols].apply(lambda row: row.value_counts().max(), axis=1)
    else:
        cluster_mode_counts = pd.Series(0, index=pred_matrix.index)

    # (c) correlation between class consensus and cluster consensus
    # compute Spearman correlation and p-value
    rho, pval = spearmanr(top_counts, cluster_mode_counts)
    # one test here, but we'll still wrap into DataFrame and apply trivial correction
    stats_df = pd.DataFrame([{
        "stat":"spearman_rho",
        "rho": float(rho),
        "p_value": float(pval)
    }])
    stats_df["p_adj_fdr_bh"] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
    stats_df.to_csv(os.path.join(RESULTS_DIR, "prediction_stability_stats.tsv"), sep="\t", index=False)
    print(f"[OK] Saved stability stats → {os.path.join(RESULTS_DIR, 'prediction_stability_stats.tsv')}")
    print(stats_df.to_string(index=False))

# ---------- DONE ----------
print("\nAll done. Outputs saved to ./results/ (predictions in ./results/predictions/).")

#!/usr/bin/env python3
"""
run_baselines.py: End-to-end training and evaluation of anomaly-detection baselines
on HDFS preprocessed data.

This version drops DeepLog placeholders and adds LocalOutlierFactor and OneClassSVM,
then plots all four continuous-score methods in the final PR curve.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, auc
)

# Number of samples to feed into DBSCAN
DBSCAN_SAMPLE_SIZE = 50000

def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate anomaly detection baselines on HDFS data"
    )
    parser.add_argument(
        "--preprocessed-dir", default="data/HDFS/preprocessed",
        help="Directory containing Event_occurrence_matrix.csv and anomaly_label.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory to save evaluation outputs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load data
    occ = pd.read_csv(os.path.join(args.preprocessed_dir, "Event_occurrence_matrix.csv"))
    labels = pd.read_csv(os.path.join(args.preprocessed_dir, "anomaly_label.csv"))
    df = occ.merge(labels, on="BlockId", suffixes=("_occ", ""))

    # 2) Features (only E* columns) and true labels
    feature_cols = [c for c in df.columns if c.startswith("E")]
    X = df[feature_cols].values
    y = df["Label"].map({"Normal": 0, "Anomaly": 1}).values

    # 3) Train/test split for supervised / score-based methods
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # -----------------------------
    # 4) RandomForest (Supervised)
    # -----------------------------
    print("\n=== RandomForest (Supervised) ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    rf_probs = rf.predict_proba(X_te)[:, 1]
    print(classification_report(y_te, y_pred_rf))
    print(f"ROC AUC: {roc_auc_score(y_te, rf_probs):.3f}")

    # ---------------------------------
    # 5) IsolationForest (Unsupervised)
    # ---------------------------------
    print("\n=== IsolationForest (Unsupervised) ===")
    iso = IsolationForest(contamination=y.mean(), random_state=42)
    iso.fit(X_tr)
    scores_iso = -iso.decision_function(X_te)
    thr_iso = np.percentile(scores_iso, 100 * (1 - y_te.mean()))
    y_pred_iso = (scores_iso >= thr_iso).astype(int)
    print(classification_report(y_te, y_pred_iso))
    print(f"ROC AUC: {roc_auc_score(y_te, scores_iso):.3f}")

    # ------------------------------------------------
    # 6) DBSCAN (Unsupervised clustering on a sample)
    # ------------------------------------------------
    # print(f"\n=== DBSCAN (Unsupervised, on {DBSCAN_SAMPLE_SIZE} random traces) ===")
    # rng = np.random.RandomState(42)
    # idx = rng.choice(X.shape[0], min(DBSCAN_SAMPLE_SIZE, X.shape[0]), replace=False)
    # X_samp, y_samp = X[idx], y[idx]
    # db = DBSCAN(eps=0.5, min_samples=10)
    # clusters = db.fit_predict(X_samp)
    # y_pred_db = (clusters == -1).astype(int)  # cluster -1 = noise = anomaly
    # print(classification_report(y_samp, y_pred_db))

    # ----------------------------------------------------
    # 7) LocalOutlierFactor (Unsupervised, score-based)
    # ----------------------------------------------------
    print("\n=== LocalOutlierFactor (Unsupervised) ===")
    mask_norm = (y_tr == 0)
    X_norm = X_tr[mask_norm]
    n_train = min(10000, X_norm.shape[0])
    idx = np.random.choice(X_norm.shape[0], n_train, replace=False)
    X_lof_train = X_norm[idx]

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=y.mean(),
        novelty=True,
        n_jobs=-1
    )
    lof.fit(X_lof_train)
    scores_lof = -lof.decision_function(X_te)
    thr_lof = np.percentile(scores_lof, 100 * (1 - y_te.mean()))
    y_pred_lof = (scores_lof >= thr_lof).astype(int)
    print(classification_report(y_te, y_pred_lof))
    print(f"ROC AUC: {roc_auc_score(y_te, scores_lof):.3f}")

    # --------------------------------------------------------
    # 8) OneClassSVM (Unsupervised, score-based)
    # --------------------------------------------------------
    print("\n=== OneClassSVM (Unsupervised) ===")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=y.mean())
    ocsvm.fit(X_tr)
    scores_ocsvm = -ocsvm.decision_function(X_te)
    thr_ocsvm = np.percentile(scores_ocsvm, 100 * (1 - y_te.mean()))
    y_pred_ocsvm = (scores_ocsvm >= thr_ocsvm).astype(int)
    print(classification_report(y_te, y_pred_ocsvm))
    print(f"ROC AUC: {roc_auc_score(y_te, scores_ocsvm):.3f}")

    # -----------------------------
    # 9) Precision-Recall Curve
    # -----------------------------
    print("\n=== Precision-Recall Curve ===")
    plt.figure(figsize=(6, 4))
    methods = {
        "RandomForest": rf_probs,
        "IsolationForest": scores_iso,
        "LocalOutlierFactor": scores_lof,
        "OneClassSVM": scores_ocsvm
    }
    for name, scores in methods.items():
        prec, rec, _ = precision_recall_curve(y_te, scores)
        pr_auc = auc(rec, prec)
        print(f"{name} PR AUC: {pr_auc:.3f}")
        plt.plot(rec, prec, label=f"{name} (AUC={pr_auc:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend()
    pr_path = os.path.join(args.output_dir, "precision_recall.png")
    plt.tight_layout()
    plt.savefig(pr_path)
    print(f"Saved Precision–Recall curve to {pr_path}")

if __name__ == "__main__":
    main()

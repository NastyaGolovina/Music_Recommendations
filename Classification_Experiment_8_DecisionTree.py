import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# =============================================================================
# Load dataset
# =============================================================================
music = pd.read_csv("music_classification.csv", dtype={
    "rank": "int16",
    "streams": "float32",
    "is_hit": "int8",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "weekday": "int8",
    "quarter": "int8",
    "is_weekend": "int8",
    "days_since_start": "int32",
    "artist_count": "int32",
})

print("music_classification.csv -> shape :", music.shape)
print("Class distribution:\n", music["is_hit"].value_counts(normalize=True).round(4))

# Sample to reduce training time
music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

# =============================================================================
# Time-based train/test split
# =============================================================================
train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = train["is_hit"]
y_test  = test["is_hit"]

print(f"\nTrain size: {len(train)} | Test size: {len(test)}")

# =============================================================================
# Target Encoding
# =============================================================================
global_mean = y_train.mean()

artist_mean = train.groupby("artist")["is_hit"].mean()
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

genre_mean = train.groupby("main_genre")["is_hit"].mean()
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = train.groupby("region")["is_hit"].mean()
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

# =============================================================================
# Feature matrix
# =============================================================================
drop_cols = ["is_hit", "artist", "main_genre", "region", "streams"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

feature_names = list(X_train.columns)

print("\n X_train -> shape :", X_train.shape)
print(" X_test  -> shape :", X_test.shape)

# =============================================================================
# MLflow — DecisionTreeClassifier with Gini analysis and max_depth sweep
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Classification")
mlflow.sklearn.autolog(log_models=False)

# Sweep over max_depth to find the optimal complexity
# Shallow trees: low variance, high bias (underfit)
# Deep trees: low bias, high variance (overfit)
depths = [3, 5, 10, 15, None]   # None = fully grown tree

best_f1    = 0
best_depth = None

for max_depth in depths:
    run_name = f"DecisionTree - max_depth={max_depth}"

    with mlflow.start_run(run_name=run_name):

        dt = DecisionTreeClassifier(
            criterion="gini",         # split quality measured by Gini impurity
            max_depth=max_depth,
            min_samples_leaf=50,
            class_weight="balanced",  # compensate for the ~7% positive class
            random_state=42,
        )

        dt.fit(X_train, y_train)
        y_pred      = dt.predict(X_test)
        y_pred_prob = dt.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        # Gini impurity of the root node (lower = purer first split)
        root_gini = dt.tree_.impurity[0]

        mlflow.log_params({
            "criterion":       "gini",
            "max_depth":       str(max_depth),
            "min_samples_leaf": 50,
            "class_weight":    "balanced",
            "sample_frac":     0.2,
            "split":           "time-based year<2021",
            "target":          "is_hit",
            "n_leaves":        dt.get_n_leaves(),
            "tree_depth":      dt.get_depth(),
        })
        mlflow.log_metrics({
            "Accuracy":   acc,
            "Precision":  precision,
            "Recall":     recall,
            "F1":         f1,
            "ROC_AUC":    roc_auc,
            "root_gini":  root_gini,
        })

        print(f"\n{run_name}")
        print(f"  Tree depth  : {dt.get_depth()}  |  Leaves: {dt.get_n_leaves()}")
        print(f"  Root Gini   : {root_gini:.4f}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  F1          : {f1:.4f}")
        print(f"  ROC-AUC     : {roc_auc:.4f}")

        # -----------------------------------------------------------------
        # Save a legible tree visualisation for the best small tree (depth=5)
        # The assignment requires drawing a legible tree and analysing Gini.
        # Deeper trees are too large to render legibly, so we limit to depth=5.
        # -----------------------------------------------------------------
        if max_depth == 5:
            fig, ax = plt.subplots(figsize=(28, 10))
            plot_tree(
                dt,
                feature_names=feature_names,
                class_names=["Not Hit", "Hit"],
                filled=True,
                rounded=True,
                fontsize=9,
                ax=ax,
                max_depth=4,          # display top 4 levels for legibility
            )
            plt.title("Decision Tree Classifier (max_depth=5) — Gini Impurity", fontsize=14)
            plt.tight_layout()
            tree_path = "decision_tree_max_depth5.png"
            plt.savefig(tree_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(tree_path)
            print(f"  Tree visualisation saved → {tree_path}")

        if f1 > best_f1:
            best_f1    = f1
            best_depth = max_depth

        print(classification_report(y_test, y_pred, target_names=["Not Hit", "Hit"]))

print(f"\nBest max_depth: {best_depth}  (F1 = {best_f1:.4f})")
print("\nAll DecisionTreeClassifier experiments completed!")


# =============================================================================
# Experiment — DecisionTreeClassifier with Gini Analysis
#
# This experiment trained a Decision Tree Classifier to predict whether a
# Spotify chart entry is a hit (is_hit = 1). The target is heavily imbalanced
# (~7% positive class), so class_weight='balanced' was applied to penalise
# misclassification of the minority class more heavily.
#
# Split quality was measured using Gini impurity. A node's Gini impurity
# equals 1 - sum(p_i^2) across classes; a pure node has Gini = 0. The root
# node's Gini value is logged for each run to track how informative the
# first split is.
#
# Five max_depth values were evaluated: 3, 5, 10, 15, and unrestricted (None).
# Shallow trees are more interpretable and less prone to overfitting but may
# miss complex patterns; deep trees capture more signal at the risk of
# memorising training data. The assignment requires a legible tree image,
# which is generated for the depth=5 model and saved as an artifact.
#
# F1-score on the positive class (Hit) is the primary evaluation metric due
# to the class imbalance. ROC-AUC provides a threshold-independent view of
# discriminative power.
# =============================================================================

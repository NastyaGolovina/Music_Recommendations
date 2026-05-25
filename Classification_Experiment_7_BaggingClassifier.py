import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

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
print(f"Train is_hit rate: {y_train.mean():.4f}")
print(f"Test  is_hit rate: {y_test.mean():.4f}")

# =============================================================================
# Target Encoding for high-cardinality categoricals
# =============================================================================
global_mean = y_train.mean()

# For classification target encoding: mean of the binary label (hit rate)
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

print("\n X_train -> shape :", X_train.shape)
print(" X_test  -> shape :", X_test.shape)

# Tree-based models do not require scaling, but we keep it consistent
# with other experiments; it does not hurt performance.

# =============================================================================
# MLflow — BaggingClassifier experiments
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Classification")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {
        "run_name":     "BaggingClassifier - n_estimators=10",
        "n_estimators": 10,
        "max_samples":  0.8,
    },
    {
        "run_name":     "BaggingClassifier - n_estimators=50",
        "n_estimators": 50,
        "max_samples":  0.8,
    },
    {
        "run_name":     "BaggingClassifier - n_estimators=100",
        "n_estimators": 100,
        "max_samples":  0.6,
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        bag_clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=50,
                random_state=42,
            ),
            n_estimators=exp["n_estimators"],
            max_samples=exp["max_samples"],
            bootstrap=True,
            random_state=42,
            n_jobs=4,
        )

        bag_clf.fit(X_train, y_train)
        y_pred      = bag_clf.predict(X_test)
        y_pred_prob = bag_clf.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_params({
            "n_estimators":   exp["n_estimators"],
            "max_samples":    exp["max_samples"],
            "bootstrap":      True,
            "base_max_depth": 10,
            "base_min_leaf":  50,
            "sample_frac":    0.2,
            "split":          "time-based year<2021",
            "target":         "is_hit",
        })
        mlflow.log_metrics({
            "Accuracy":  acc,
            "Precision": precision,
            "Recall":    recall,
            "F1":        f1,
            "ROC_AUC":   roc_auc,
        })

        print(f"\n{exp['run_name']}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Not Hit", "Hit"]))

print("\nAll BaggingClassifier experiments completed!")


# =============================================================================
# Experiment — BaggingClassifier
#
# This experiment evaluated a Bagging Classifier for predicting whether a
# Spotify chart entry would be a "hit" (is_hit = 1). The is_hit label was
# defined in the data preparation notebook based on a stream count threshold.
# The dataset was sampled to 20% and split temporally (before 2021 for
# training, 2021 onward for testing).
#
# Target encoding was applied to the three high-cardinality categorical
# columns (artist, genre, region), using the mean hit rate computed on
# the training set, with unseen categories falling back to the global mean.
#
# Three ensemble sizes were evaluated: 10, 50, and 100 Decision Tree base
# learners. Each tree was limited to a max depth of 10 and minimum 50 samples
# per leaf to prevent overfitting at the base estimator level.
#
# The is_hit target is heavily imbalanced (~7% positive class), which means
# accuracy alone is misleading — Precision, Recall, F1, and ROC-AUC are the
# primary metrics. Bagging reduces variance without changing the individual
# tree's inability to handle class imbalance; if recall on the positive class
# is low, techniques like class_weight='balanced' or oversampling could be
# explored in future iterations.
# =============================================================================

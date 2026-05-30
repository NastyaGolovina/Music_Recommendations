import os
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import (
    StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
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

music = music.sample(frac=0.1, random_state=42).reset_index(drop=True)
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

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print("\n X_train_scaled -> shape :", X_train_scaled.shape)
print(" X_test_scaled  -> shape :", X_test_scaled.shape)

# =============================================================================
# MLflow
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Classification")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {
        "run_name": "Stacking - LR + DT + RF -> LR",
        "estimators": [
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=5,
                                          n_jobs=4, random_state=42)),
        ],
        "final_estimator": LogisticRegression(max_iter=500, random_state=42),
    },
    {
        "run_name": "Stacking - DT + RF + GB -> LR",
        "estimators": [
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=5,
                                          n_jobs=4, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                              random_state=42)),
        ],
        "final_estimator": LogisticRegression(max_iter=500, random_state=42),
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        stack_clf = StackingClassifier(
            estimators=exp["estimators"],
            final_estimator=exp["final_estimator"],
            n_jobs=4,
            passthrough=False,
        )

        stack_clf.fit(X_train_scaled, y_train)
        y_pred      = stack_clf.predict(X_test_scaled)
        y_pred_prob = stack_clf.predict_proba(X_test_scaled)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        # Log apenas as métricas — o autolog já guardou os parâmetros
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

print("\nAll StackingClassifier experiments completed!")


# =============================================================================
# Experimento — StackingClassifier
#
# O Stacking combina modelos DIFERENTES e usa um meta-modelo para aprender
# a melhor forma de combinar as suas previsoes. Ao contrario do Bagging
# (mesmo modelo, dados diferentes) e do Boosting (mesmo modelo, sequencial),
# o Stacking usa diversidade de modelos para capturar diferentes padroes.
#
# Nivel 1 (base models): varios modelos diferentes treinam nos dados
# Nivel 2 (meta-model): aprende a combinar as previsoes dos modelos base
#
# Foram testadas 2 configuracoes com diferentes combinacoes de base models.
# StandardScaler foi aplicado porque LogisticRegression e sensivel a escala.
# =============================================================================

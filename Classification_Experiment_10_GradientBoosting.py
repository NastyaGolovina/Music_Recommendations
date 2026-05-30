import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
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

# Sample to reduce training time — GradientBoosting is slow (sequential training)
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

print("\n X_train -> shape :", X_train.shape)
print(" X_test  -> shape :", X_test.shape)

# =============================================================================
# MLflow — GradientBoostingClassifier experiments
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Classification")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {
        "run_name":      "GradientBoosting - n=100 lr=0.1",
        "n_estimators":  100,
        "learning_rate": 0.1,
        "max_depth":     3,
    },
    {
        "run_name":      "GradientBoosting - n=100 lr=0.05",
        "n_estimators":  100,
        "learning_rate": 0.05,
        "max_depth":     3,
    },
    {
        "run_name":      "GradientBoosting - n=200 lr=0.1",
        "n_estimators":  200,
        "learning_rate": 0.1,
        "max_depth":     3,
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        gb_clf = GradientBoostingClassifier(
            n_estimators=exp["n_estimators"],
            learning_rate=exp["learning_rate"],
            max_depth=exp["max_depth"],
            random_state=42,
        )

        gb_clf.fit(X_train, y_train)
        y_pred      = gb_clf.predict(X_test)
        y_pred_prob = gb_clf.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_params({
            "n_estimators":  exp["n_estimators"],
            "learning_rate": exp["learning_rate"],
            "max_depth":     exp["max_depth"],
            "sample_frac":   0.1,
            "split":         "time-based year<2021",
            "target":        "is_hit",
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

print("\nAll GradientBoostingClassifier experiments completed!")


# =============================================================================
# Experimento — GradientBoostingClassifier
#
# Este experimento avaliou o Gradient Boosting para classificar músicas do
# Spotify como hits ou não-hits. Ao contrário do Bagging que treina árvores
# em paralelo, o Gradient Boosting treina árvores sequencialmente — cada
# árvore nova foca-se nos erros da anterior, usando o gradiente para
# minimizar a função de custo.
#
# O dataset foi amostrado a 10% (GradientBoosting é lento por natureza
# sequencial) e dividido temporalmente, com treino antes de 2021 e teste
# a partir de 2021.
#
# Foram testadas 3 configurações variando n_estimators e learning_rate:
#   - learning_rate controla o tamanho do passo de cada correção
#   - n_estimators controla quantas árvores sequenciais são treinadas
#
# O target is_hit é desequilibrado (~7% positivo). As métricas principais
# são F1 e ROC-AUC. Espera-se que o GradientBoosting supere o
# BaggingClassifier devido ao seu mecanismo de correção sequencial.
# =============================================================================

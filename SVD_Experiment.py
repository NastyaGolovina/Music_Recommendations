import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
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

music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
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

drop_cols = ["is_hit", "artist", "main_genre", "region", "streams"]
X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

n_original_features = X_train.shape[1]
print(f"\nOriginal features: {n_original_features}")

# Scale before SVD
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

# =============================================================================
# SVD — Singular Value Decomposition (dimensionality reduction)
# =============================================================================
# SVD decomposes the data matrix X into U × Σ × Vᵀ
# TruncatedSVD keeps only the top k singular values/vectors
# This is equivalent to PCA but works directly on the matrix
# without centering (already done by StandardScaler)

print("\nApplying TruncatedSVD with different n_components...")

svd_components = [3, 5, 8, 10, 12]
explained_variance = []

for n in svd_components:
    svd = TruncatedSVD(n_components=n, random_state=42)
    svd.fit(X_train_scaled)
    var = svd.explained_variance_ratio_.sum()
    explained_variance.append(var)
    print(f"  n_components={n:2d}  explained_variance={var:.4f} ({var*100:.1f}%)")

# Plot explained variance
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(svd_components, [v * 100 for v in explained_variance],
        marker="o", color="steelblue", linewidth=2)
ax.axhline(y=90, color="red", linestyle="--", label="90% threshold")
ax.set_title("SVD — Variância Explicada por Número de Componentes", fontsize=13)
ax.set_xlabel("Número de componentes SVD", fontsize=11)
ax.set_ylabel("Variância Explicada (%)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
svd_variance_path = "svd_explained_variance.png"
plt.savefig(svd_variance_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSVD variance plot saved → {svd_variance_path}")

# =============================================================================
# Compare models WITH and WITHOUT SVD
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - SVD")

models = [
    ("LogisticRegression",
     LogisticRegression(max_iter=500, random_state=42)),
    ("DecisionTree",
     DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("RandomForest",
     RandomForestClassifier(n_estimators=50, max_depth=5,
                            n_jobs=4, random_state=42)),
]

# Choose n_components that explains ~85%+ variance
best_n = svd_components[
    next(i for i, v in enumerate(explained_variance) if v >= 0.85)
    if any(v >= 0.85 for v in explained_variance)
    else -1
]
print(f"\nUsing n_components={best_n} for model comparison")

svd = TruncatedSVD(n_components=best_n, random_state=42)
X_train_svd = svd.fit_transform(X_train_scaled)
X_test_svd  = svd.transform(X_test_scaled)

print(f"Shape after SVD: train={X_train_svd.shape}, test={X_test_svd.shape}")

results = []

for model_name, model in models:
    # --- Without SVD ---
    with mlflow.start_run(run_name=f"{model_name} - Without SVD"):
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        fit_time = time.time() - t0

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({
            "model":        model_name,
            "svd_applied":  False,
            "n_features":   n_original_features,
            "sample_frac":  0.05,
        })
        mlflow.log_metrics({
            "Accuracy": acc, "F1": f1,
            "ROC_AUC": roc_auc, "fit_time_seconds": fit_time,
        })

        results.append({
            "model": model_name, "svd": "No SVD",
            "F1": f1, "ROC_AUC": roc_auc, "fit_time": fit_time,
            "n_features": n_original_features,
        })
        print(f"\n{model_name} — Without SVD")
        print(f"  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  fit_time={fit_time:.2f}s")

    # --- With SVD ---
    with mlflow.start_run(run_name=f"{model_name} - With SVD ({best_n} components)"):
        t0 = time.time()
        model.fit(X_train_svd, y_train)
        fit_time_svd = time.time() - t0

        y_pred_svd = model.predict(X_test_svd)
        y_prob_svd = model.predict_proba(X_test_svd)[:, 1]

        acc_svd     = accuracy_score(y_test, y_pred_svd)
        f1_svd      = f1_score(y_test, y_pred_svd, zero_division=0)
        roc_auc_svd = roc_auc_score(y_test, y_prob_svd)

        mlflow.log_params({
            "model":          model_name,
            "svd_applied":    True,
            "n_components":   best_n,
            "n_features_orig": n_original_features,
            "explained_var":  round(svd.explained_variance_ratio_.sum(), 4),
            "sample_frac":    0.05,
        })
        mlflow.log_metrics({
            "Accuracy": acc_svd, "F1": f1_svd,
            "ROC_AUC": roc_auc_svd, "fit_time_seconds": fit_time_svd,
        })
        mlflow.log_artifact(svd_variance_path)

        results.append({
            "model": model_name, "svd": f"SVD ({best_n})",
            "F1": f1_svd, "ROC_AUC": roc_auc_svd, "fit_time": fit_time_svd,
            "n_features": best_n,
        })
        print(f"{model_name} — With SVD ({best_n} components)")
        print(f"  F1={f1_svd:.4f}  ROC-AUC={roc_auc_svd:.4f}  fit_time={fit_time_svd:.2f}s")

# =============================================================================
# Summary plot — F1 and fit time comparison
# =============================================================================
df_results = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# F1 comparison
models_list  = df_results["model"].unique()
x            = np.arange(len(models_list))
width        = 0.35

ax = axes[0]
no_svd_f1  = df_results[df_results["svd"] == "No SVD"]["F1"].values
svd_f1     = df_results[df_results["svd"] != "No SVD"]["F1"].values
ax.bar(x - width/2, no_svd_f1, width, label="Sem SVD", color="steelblue")
ax.bar(x + width/2, svd_f1,    width, label=f"Com SVD ({best_n} comp.)", color="coral")
ax.set_title("F1 Score — Com vs Sem SVD", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15)
ax.set_ylabel("F1 Score")
ax.legend()

# Fit time comparison
ax2 = axes[1]
no_svd_time = df_results[df_results["svd"] == "No SVD"]["fit_time"].values
svd_time    = df_results[df_results["svd"] != "No SVD"]["fit_time"].values
ax2.bar(x - width/2, no_svd_time, width, label="Sem SVD", color="steelblue")
ax2.bar(x + width/2, svd_time,    width, label=f"Com SVD ({best_n} comp.)", color="coral")
ax2.set_title("Tempo de Treino — Com vs Sem SVD", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models_list, rotation=15)
ax2.set_ylabel("Tempo (segundos)")
ax2.legend()

plt.suptitle("Impacto do SVD na Performance e Velocidade dos Modelos", fontsize=13)
plt.tight_layout()
comparison_path = "svd_model_comparison.png"
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nComparison plot saved → {comparison_path}")
print("\nAll SVD experiments completed!")


# =============================================================================
# Experimento — SVD (Singular Value Decomposition)
#
# O SVD é uma técnica de redução de dimensionalidade que decompõe a
# matriz de dados X em três matrizes: U × Σ × Vᵀ. O TruncatedSVD
# mantém apenas os k maiores valores singulares, que capturam a maior
# parte da variância dos dados com muito menos features.
#
# Isto é equivalente ao PCA mas trabalha diretamente na matriz sem
# necessitar de centrar os dados (o StandardScaler já fez isso).
#
# O objetivo deste experimento é comparar o desempenho e o tempo de
# treino dos modelos COM e SEM redução de dimensionalidade por SVD:
# - Com menos features, os modelos treinam mais rápido
# - A perda de performance deve ser mínima se o SVD capturar
#   variância suficiente (objetivo: >= 85%)
#
# Foram testados diferentes números de componentes SVD para encontrar
# o mínimo que mantém >= 85% da variância explicada. Depois comparámos
# LogisticRegression, DecisionTree e RandomForest com e sem SVD,
# analisando F1, ROC-AUC e tempo de treino.
# =============================================================================

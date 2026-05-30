import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# =============================================================================
# Load dataset
# =============================================================================
music = pd.read_csv("music_regression.csv", dtype={
    "rank": "int16",
    "streams": "float32",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "weekday": "int8",
    "quarter": "int8",
    "is_weekend": "int8",
    "days_since_start": "int32",
    "artist_count": "int32",
})

print("music_regression.csv -> shape :", music.shape)

# Hierarchical clustering builds a full O(n²) distance matrix
# 3000 samples → 3000×3000/2 = 4.5M distances → fits in memory
music = music.sample(n=3000, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

# =============================================================================
# Feature preparation
# =============================================================================
global_mean = np.log1p(music["streams"]).mean()

artist_mean = music.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
music["artist_te"] = music["artist"].map(artist_mean).fillna(global_mean)

genre_mean = music.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
music["genre_te"] = music["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = music.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
music["region_te"] = music["region"].map(region_mean).fillna(global_mean)

feature_cols = [
    "rank", "year", "month", "weekday", "quarter", "is_weekend",
    "days_since_start", "artist_count", "artist_te", "genre_te", "region_te"
]

X = music[feature_cols].copy()

# Scaling is essential — clustering uses distances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features for clustering: {feature_cols}")
print(f"X_scaled shape: {X_scaled.shape}")

# =============================================================================
# MLflow setup
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Clustering")

# =============================================================================
# Step 1 — Dendrogram
# =============================================================================
print("\nComputing linkage matrix for dendrogram...")

Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax,
    color_threshold=0,
)
ax.set_title("Hierarchical Clustering — Dendrograma (Ward Linkage)", fontsize=14)
ax.set_xlabel("Amostras (ou tamanho do cluster)", fontsize=11)
ax.set_ylabel("Distância (Ward)", fontsize=11)
ax.axhline(y=15, color="red",    linestyle="--", linewidth=1.5, label="Threshold k=3")
ax.axhline(y=10, color="orange", linestyle="--", linewidth=1.5, label="Threshold k=5")
ax.legend(fontsize=10)
plt.tight_layout()
dendrogram_path = "hierarchical_dendrogram.png"
plt.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Dendrograma guardado → {dendrogram_path}")

# =============================================================================
# Step 2 — Silhouette analysis k=2 to 8
# =============================================================================
print("\nSilhouette analysis for k = 2 to 8...")

k_values          = list(range(2, 9))
silhouette_scores = []

for k in k_values:
    labels = fcluster(Z, k, criterion="maxclust")
    score  = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}  silhouette={score:.4f}")

best_k = k_values[int(np.argmax(silhouette_scores))]
print(f"\nBest k (highest silhouette): {best_k}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_values, silhouette_scores, marker="o", color="steelblue", linewidth=2)
ax.axvline(x=best_k, color="red", linestyle="--", label=f"Best k={best_k}")
ax.set_title("Silhouette Score por número de clusters", fontsize=13)
ax.set_xlabel("Número de clusters (k)", fontsize=11)
ax.set_ylabel("Silhouette Score", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
silhouette_path = "hierarchical_silhouette.png"
plt.savefig(silhouette_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Silhouette plot guardado → {silhouette_path}")

# =============================================================================
# Step 3 — Final clustering with MLflow logging
# =============================================================================
for k in sorted(set([best_k, 3, 5])):
    run_name = f"HierarchicalClustering - k={k} - ward"
    with mlflow.start_run(run_name=run_name):

        model  = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes  = dict(zip(unique.tolist(), counts.tolist()))

        mlflow.log_params({
            "n_clusters":  k,
            "linkage":     "ward",
            "n_samples":   3000,
            "n_features":  len(feature_cols),
            "scaler":      "StandardScaler",
        })
        mlflow.log_metrics({"silhouette_score": sil_score})
        mlflow.log_artifact(dendrogram_path)
        mlflow.log_artifact(silhouette_path)

        print(f"\nk={k}  silhouette={sil_score:.4f}")
        print(f"  Cluster sizes: {cluster_sizes}")

print("\nAll Hierarchical Clustering experiments completed!")


# =============================================================================
# Experimento — Hierarchical Clustering (Clustering Hierárquico)
#
# O clustering hierárquico aglomerativo começa com cada ponto no seu
# próprio cluster e vai juntando os mais próximos progressivamente.
# Esta história é visualizada no DENDROGRAMA — mostra quando e a que
# distância os clusters se fundem.
#
# Usámos Ward linkage — minimiza a variância total dentro dos clusters,
# produzindo clusters compactos e bem separados.
#
# O número ótimo de clusters foi determinado por:
# 1. Análise visual do dendrograma (maior salto de distância)
# 2. Silhouette Score — mede quão bem cada ponto está no seu cluster
#    (1=perfeito, 0=fronteira, -1=errado)
#
# Usámos 3000 amostras porque o clustering hierárquico constrói uma
# matriz de distâncias O(n²) — com mais amostras a memória RAM esgota.
# =============================================================================

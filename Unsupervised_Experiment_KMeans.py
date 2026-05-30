import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

music = pd.read_csv("music_classification.csv", dtype={
    "rank":             "int16",
    "streams":          "float32",
    "is_hit":           "int8",
    "year":             "int16",
    "month":            "int8",
    "day":              "int8",
    "weekday":          "int8",
    "quarter":          "int8",
    "is_weekend":       "int8",
    "days_since_start": "int32",
    "artist_count":     "int32",
})

print("music_classification.csv -> shape :", music.shape)

music = music.sample(frac=0.001, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

global_mean = music["is_hit"].mean()

artist_mean = music.groupby("artist")["is_hit"].mean()
music["artist_te"] = music["artist"].map(artist_mean).fillna(global_mean)

genre_mean = music.groupby("main_genre")["is_hit"].mean()
music["genre_te"] = music["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = music.groupby("region")["is_hit"].mean()
music["region_te"] = music["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]
X = music.drop(columns=drop_cols)

print("X -> shape :", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

pca = PCA(n_components=11, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"X after PCA -> shape : {X_pca.shape}")

results = []
for k in [2, 3]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    results.append({"k": k, "silhouette": sil})
    print(f"  k={k} -> Silhouette Score: {sil:.4f}")

best = max(results, key=lambda x: x["silhouette"])
print(f"\nBest k: {best['k']} with Silhouette Score: {best['silhouette']:.4f}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - KMeans Clustering")

with mlflow.start_run(run_name="Experiment - KMeans"):
    mlflow.log_param("sample_frac",    0.001)
    mlflow.log_param("pca_components", 11)
    mlflow.log_param("k_values_tried", "2,3")
    mlflow.log_param("best_k",         best["k"])
    mlflow.log_metric("best_silhouette_score", best["silhouette"])

    for r in results:
        mlflow.log_metric(f"silhouette_k{r['k']}", r["silhouette"])

    print(f"\nExperiment - KMeans Clustering")
    for r in results:
        print(f"  k={r['k']} Silhouette: {r['silhouette']:.4f}")
    print(f"  Best k: {best['k']}")
    print(f"  Best Silhouette Score: {best['silhouette']:.4f}")

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - KMeans Clustering"])
runs.to_csv("runs/Spotify_Streams_KMeans_runs.csv", index=False)
print("CSV saved!")


# ## Experiment — KMeans Clustering
#
# This experiment applied KMeans clustering to uncover natural groupings in
# the Spotify dataset without using the target label (is_hit). The data was
# first scaled and reduced using PCA (11 components, ~95% variance retained)
# to improve clustering stability and reduce noise.
#
# KMeans was evaluated using silhouette score for k=2 and k=3. The best result
# was obtained with k=2, achieving a silhouette score of 0.1364.
#
# Although k=2 performed slightly better, the overall silhouette scores were
# low, indicating weak cluster separation. This suggests that hit songs and
# non-hit songs do not naturally form distinct groups in feature space, likely
# due to overlapping influences such as artist popularity, region, and timing.
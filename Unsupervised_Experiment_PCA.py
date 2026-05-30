import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

# Target encoding
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

# PCA requires scaling — features must be on same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Now run PCA with optimal components
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"X after PCA -> shape : {X_pca.shape}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - PCA")

with mlflow.start_run(run_name="Experiment - PCA"):
    mlflow.log_param("sample_frac",        0.05)
    mlflow.log_param("original_features",  X_scaled.shape[1])
    mlflow.log_param("n_components",       n_components_95)
    mlflow.log_param("variance_threshold", 0.95)
    mlflow.log_metric("total_variance_explained", pca.explained_variance_ratio_.sum())
    mlflow.log_metric("n_components_95",          n_components_95)

    for i, var in enumerate(pca.explained_variance_ratio_):
        mlflow.log_metric(f"variance_pc{i+1}", var)

    print(f"\nExperiment - PCA")
    print(f"  Original features    : {X_scaled.shape[1]}")
    print(f"  Components for 95%   : {n_components_95}")
    print(f"  Variance explained   : {pca.explained_variance_ratio_.sum():.4f}")
    print(f"\nVariance per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - PCA"])
runs.to_csv("runs/Spotify_Streams_PCA_runs.csv", index=False)
print("CSV saved!")

# ## Experiment — PCA (Principal Component Analysis)
#
# This experiment applied Principal Component Analysis (PCA) to reduce the
# dimensionality of the Spotify dataset while retaining 95% of the original
# variance. Artist, genre, and region were target encoded, and StandardScaler
# was applied before PCA because the technique is sensitive to feature scale.
#
# PCA reduced the feature space from 16 variables to 11 principal components
# while preserving approximately 95.7% of the total variance. This indicates
# that some redundancy exists among the original features, although the variance
# remains distributed across multiple components.
#
# The reduced feature representation will be used as input for subsequent
# clustering experiments, improving efficiency while preserving most of the
# information contained in the dataset.
import pandas as pd
import numpy as np
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

# sample to reduce training time
music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"].values)
y_test = np.log1p(test["streams"].values)

global_mean = y_train.mean()

# target encoding
artist_mean = train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

genre_mean = train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "artist", "main_genre", "region"]
X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

# KNN requires scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print(" X_test_scaled  -> shape :", X_test_scaled.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - KNeighborsRegressor")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "KNN - k=1",  "n_neighbors": 1},
    {"run_name": "KNN - k=3",  "n_neighbors": 3},
    {"run_name": "KNN - k=5",  "n_neighbors": 5},
    {"run_name": "KNN - k=10", "n_neighbors": 10},
    {"run_name": "KNN - k=15", "n_neighbors": 15},
    {"run_name": "KNN - k=20", "n_neighbors": 20},
]

k_values, rmse_values = [], []

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        knn = KNeighborsRegressor(
            n_neighbors=exp["n_neighbors"],
            n_jobs=-1
        )
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "n_neighbors": exp["n_neighbors"],
            "split":       "time-based year<2021",
            "target":      "log1p(streams)",
            "sample_frac": 0.05,
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)

        k_values.append(exp["n_neighbors"])
        rmse_values.append(rmse)

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

# Elbow curve — RMSE vs k
best_k = k_values[int(np.argmin(rmse_values))]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, rmse_values, marker="o", color="steelblue", linewidth=2)
ax.axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={best_k}")
ax.set_title("KNN Regressor — RMSE vs Number of Neighbours")
ax.set_xlabel("k (n_neighbors)")
ax.set_ylabel("RMSE")
ax.legend()
plt.tight_layout()
plot_path = "runs/png/KNeighborsRegressor_elbow.png"
plt.savefig(plot_path, dpi=150)
plt.close()

with mlflow.start_run(run_name="KNN - elbow curve"):
    mlflow.log_artifact(plot_path)

print(f"Elbow curve saved → {plot_path} | Optimal k={best_k}")

# =============================================================================
# Experiment 4 — KNeighbors Regressor
# =============================================================================
#
# Goal:
#   Predict log1p(streams) using K-Nearest Neighbors Regression. KNN predicts
#   by averaging the target values of the k closest training samples in feature
#   space. Since KNN is sensitive to feature scale, StandardScaler was applied
#   before training. Six values of k were tested to identify the optimal number
#   of neighbours using an elbow curve of RMSE vs k.
#
# Setup:
#   - 5% sample of music_regression.csv (time-based split: train < 2021, test >= 2021)
#   - Target encoding applied to artist, genre, region
#   - StandardScaler applied before training
#
# Results:
#   k    RMSE      R²      Train R²   Duration
#   1    0.9031    0.5589  0.9998     15.0min
#   3    0.7756    0.6747  0.8888     15.8min
#   5    0.7477    0.6976  0.8645     16.1min
#   10   0.7274    0.7139  0.8425     16.0min
#   15   0.7233    0.7170  0.8326     16.3min
#   20   0.7221    0.7180  0.8263     17.5min
#
# Analysis:
#   k=1 severely overfits — training R²=0.9998 means the model memorises the
#   training data exactly, but generalises poorly (test R²=0.5589, RMSE=0.9031).
#   As k increases, overfitting reduces and test performance improves steadily.
#   The gains plateau between k=15 and k=20 (RMSE 0.7233 vs 0.7221), making
#   k=20 the optimal choice with the best test RMSE (0.7221) and R² (0.7180).
#   All runs took roughly the same time (~15-17min) regardless of k, confirming
#   that KNN's cost is dominated by prediction (computing distances to all
#   training points) rather than the choice of k itself.
# =============================================================================
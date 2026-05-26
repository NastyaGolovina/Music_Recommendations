import pandas as pd
import numpy as np
import mlflow.sklearn
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
    {"run_name": "KNN - k=5",  "n_neighbors": 5},
    {"run_name": "KNN - k=10", "n_neighbors": 10},
    {"run_name": "KNN - k=20", "n_neighbors": 20},
]

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

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

# ## Experiment 4 — KNeighbors Regressor
#
# This experiment applied K-Nearest Neighbors regression to predict log1p(streams).
# KNN is a non-parametric model that predicts based on the average of the k closest
# training samples in feature space. Since KNN is sensitive to feature scale,
# StandardScaler was applied before training. Three values of k were tested (5, 10, 20)
# to find the optimal number of neighbours. KNN is expected to capture local patterns
# in the data but may struggle with the large dataset size due to its high computational
# cost at prediction time.
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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
music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# target encoding for artist, genre, region
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

print(" X_train -> shape :", X_train.shape)
print(" X_test  -> shape :", X_test.shape)

# Random Forest does not require scaling
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - RandomForestRegressor")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "RandomForest - n=50",  "n_estimators": 50},
    {"run_name": "RandomForest - n=100", "n_estimators": 100},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        rf = RandomForestRegressor(
            n_estimators=exp["n_estimators"],
            max_depth=10,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "n_estimators":     exp["n_estimators"],
            "max_depth":        10,
            "min_samples_leaf": 50,
            "split":            "time-based year<2021",
            "target":           "log1p(streams)",
            "sample_frac":      0.2,
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

mlflow.sklearn.autolog(disable=True)

fig, ax = plt.subplots(figsize=(30, 10))
plot_tree(rf.estimators_[0], feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=8, ax=ax,max_depth=4)
plt.title("Random Forest — Single Tree Estimator (estimators_[0])")
plt.tight_layout()
plt.savefig("runs/png/RandomForestRegressor_tree.png", dpi=150)
plt.close()

with mlflow.start_run(run_name="RandomForest - tree visualization"):
    mlflow.log_artifact("runs/png/RandomForestRegressor_tree.png")

print("Random Forest tree visualization saved.")

# =============================================================================
# Experiment 6 — Random Forest Regressor
# =============================================================================
#
# Goal:
#   Predict log1p(streams) using a Random Forest Regressor. Random Forest is an
#   ensemble method that builds multiple independent decision trees and averages
#   their predictions, reducing overfitting compared to a single Decision Tree.
#
# Setup:
#   - 20% sample of music_regression.csv (time-based split: train < 2021, test >= 2021)
#   - Target encoding applied to artist, genre, region
#   - max_depth=10, min_samples_leaf=50 fixed across all runs
#   - n_jobs=-1 to use all available CPU cores
#
# Results:
#   Run              RMSE      R²      Train R²   Duration
#   n_estimators=50  0.6141    0.7956  0.8607     2.6min
#   n_estimators=100 0.6141    0.7956  0.8608     5.3min
#
# Analysis:
#   Both configurations produce nearly identical RMSE and R², meaning doubling
#   the number of trees (50 → 100) gives no meaningful accuracy gain while
#   doubling training time. n_estimators=50 is therefore the more efficient choice.
#   Compared to the single Decision Tree (max_depth=10, RMSE=0.6220), Random Forest
#   improves RMSE slightly (0.6141) with better generalisation due to averaging.
#   One tree from the forest was visualised (max_depth=4) and saved as a PNG.
# =============================================================================
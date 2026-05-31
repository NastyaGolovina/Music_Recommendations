import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - AdaBoostRegressor")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "AdaBoost - n=50",  "n_estimators": 50,  "learning_rate": 0.1},
    {"run_name": "AdaBoost - n=100", "n_estimators": 100, "learning_rate": 0.1},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5, random_state=42),
            n_estimators=exp["n_estimators"],
            learning_rate=exp["learning_rate"],
            random_state=42
        )
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "n_estimators":  exp["n_estimators"],
            "learning_rate": exp["learning_rate"],
            "base_estimator": "DecisionTree max_depth=5",
            "split":         "time-based year<2021",
            "target":        "log1p(streams)",
            "sample_frac":   0.2,
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

# Experiment 7 — AdaBoost Regressor
#
# Goal:
#   Predict log1p(streams) using AdaBoost Regressor. Unlike Random Forest which
#   builds trees independently, AdaBoost trains trees sequentially — each new
#   tree focuses more on the samples the previous trees got wrong, gradually
#   correcting errors. A shallow DecisionTreeRegressor (max_depth=5) is used
#   as the base estimator (weak learner).
#
# Setup:
#   - 20% sample of music_regression.csv (time-based split: train < 2021, test >= 2021)
#   - Target encoding applied to artist, genre, region
#   - Base estimator: DecisionTreeRegressor(max_depth=5)
#   - learning_rate=0.1 fixed across all runs
#
# Results:
#   Run              RMSE      R²      Train R²   Duration
#   n_estimators=50  0.7270    0.7135  0.7951     9.7min
#   n_estimators=100 0.7358    0.7066  0.7907     23.7min
#
# Analysis:
#   Surprisingly, n_estimators=50 outperforms n_estimators=100 on both RMSE
#   (0.7270 vs 0.7358) and R² (0.7135 vs 0.7066), suggesting the model begins
#   to overfit after 50 trees with this learning rate. Training time more than
#   doubles (9.7min → 23.7min) for a worse result, making n_estimators=50 the
#   better choice. Compared to Random Forest (RMSE=0.6141), AdaBoost performs
#   noticeably worse on this dataset, likely because sequential boosting is more
#   sensitive to noisy data such as stream counts.

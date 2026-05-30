import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

music = pd.read_csv("music_regression.csv", dtype={
    "rank":             "int16",
    "streams":          "float32",
    "year":             "int16",
    "month":            "int8",
    "day":              "int8",
    "weekday":          "int8",
    "quarter":          "int8",
    "is_weekend":       "int8",
    "days_since_start": "int32",
    "artist_count":     "int32",
})

print("music_regression.csv -> shape :", music.shape)

music = music.sample(frac=0.002, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# Artist → average log1p(streams) per artist
artist_mean = train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# Genre → average log1p(streams) per genre
genre_mean = train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# Region → average log1p(streams) per region
region_mean = train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test   -> shape :", X_test.shape)

# SVR is extremely sensitive to feature scale — StandardScaler is mandatory
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print("X_test_scaled   -> shape :", X_test_scaled.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - SVR Regression")
mlflow.sklearn.autolog(log_models=False)

param_grid = [
    {"C": [0.1, 1.0], "kernel": ["rbf"], "gamma": ["scale"]},
]

grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
print(f"Best CV score   (GridSearchCV): {grid.best_score_:.4f}")

with mlflow.start_run(run_name="Experiment 12 - GridSearchCV SVR"):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)


    mlflow.log_param("model_type",     "SVR")
    mlflow.log_param("split",          "time-based year<2021")
    mlflow.log_param("sample_frac",    0.002)
    mlflow.log_param("C",              grid.best_params_["C"])
    mlflow.log_param("kernel",         grid.best_params_["kernel"])
    mlflow.log_param("gamma",          grid.best_params_["gamma"])
    mlflow.log_param("target",         "log1p(streams)")
    mlflow.log_metric("RMSE",          rmse)
    mlflow.log_metric("R2",            r2)
    mlflow.log_metric("Best_CV_score", grid.best_score_)

    print(f"\nExperiment 12 - GridSearchCV SVR")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - SVR Regression"])
runs.to_csv("runs/Spotify_Streams_SVR_runs.csv", index=False)
print("CSV saved!")

# ## Experiment 12 — SVR Regression
#
# This experiment applied Support Vector Regression (SVR) to predict Spotify
# stream counts using a log-transformed target variable. Artist, genre, and
# region were target encoded, while StandardScaler was applied because SVR is
# highly sensitive to feature scale.
#
# A time-based split was used, with data before 2021 for training and data from
# 2021 onwards for testing. Due to the computational cost of SVR, only 0.2% of
# the dataset was used. GridSearchCV was used to tune the model parameters.
#
# The best configuration achieved an RMSE of 0.7675 and an R² score of 0.6818.
# Performance was slightly better than Linear Regression and Ridge Regression,
# indicating that the RBF kernel was able to capture some non-linear patterns
# present in the data.
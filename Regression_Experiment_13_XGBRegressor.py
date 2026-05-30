import pandas as pd
import numpy as np
import mlflow.sklearn
from xgboost import XGBRegressor
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

music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
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

# XGBoost does not require scaling
drop_cols = ["streams", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test   -> shape :", X_test.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - XGBRegressor")
mlflow.sklearn.autolog(log_models=False)

param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [4, 6],
    "learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(
    estimator=XGBRegressor(
        tree_method="hist",
        random_state=42,
        n_jobs=4,
        verbosity=0,
    ),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
print(f"Best CV score   (GridSearchCV): {grid.best_score_:.4f}")

with mlflow.start_run(run_name="Experiment 13 - GridSearchCV XGBRegressor"):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    mlflow.log_param("model_type",    "XGBRegressor")
    mlflow.log_param("split",         "time-based year<2021")
    mlflow.log_param("sample_frac",   0.05)
    mlflow.log_param("n_estimators",  grid.best_params_["n_estimators"])
    mlflow.log_param("max_depth",     grid.best_params_["max_depth"])
    mlflow.log_param("learning_rate", grid.best_params_["learning_rate"])
    mlflow.log_param("target",        "log1p(streams)")
    mlflow.log_metric("RMSE",         rmse)
    mlflow.log_metric("R2",           r2)
    mlflow.log_metric("Best_CV_score", grid.best_score_)

    print(f"\nExperiment 13 - GridSearchCV XGBRegressor")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - XGBRegressor"])
runs.to_csv("runs/Spotify_Streams_XGBRegressor_runs.csv", index=False)
print("CSV saved!")


# ## Experiment — XGBRegressor
#
# This experiment applied XGBoost Regression to predict Spotify stream counts
# using a log-transformed target variable. Artist, genre, and region were
# target encoded, while no feature scaling was required because XGBoost is a
# tree-based algorithm.
#
# A time-based split was used, with data before 2021 for training and data from
# 2021 onwards for testing. GridSearchCV was used to tune the number of trees,
# tree depth, and learning rate.
#
# The best model achieved an RMSE of 0.5805 and an R² score of 0.8177,
# substantially outperforming Linear Regression and Ridge Regression. The
# results demonstrate XGBoost's ability to capture complex non-linear
# relationships within the streaming data.
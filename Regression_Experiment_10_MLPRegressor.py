import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

# Sample to reduce training time (MLP is slow on 26M rows)
music = music.sample(frac=0.1, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

# =============================================================================
# Time-based train/test split
# =============================================================================
train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# =============================================================================
# Target Encoding
# =============================================================================
artist_mean = train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

genre_mean = train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

# =============================================================================
# Feature matrix
# =============================================================================
drop_cols = ["streams", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

# Neural networks require scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print(" X_test_scaled  -> shape :", X_test_scaled.shape)

# =============================================================================
# MLflow — MLPRegressor experiments
# Single-layer and multi-layer architectures
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Regression")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {
        "run_name":     "MLP - Single Layer (64)",
        "hidden_layers": (64,),
        "activation":   "relu",
        "learning_rate_init": 0.001,
        "max_iter":     200,
    },
    {
        "run_name":     "MLP - Single Layer (128)",
        "hidden_layers": (128,),
        "activation":   "relu",
        "learning_rate_init": 0.001,
        "max_iter":     200,
    },
    {
        "run_name":     "MLP - Multi Layer (128, 64)",
        "hidden_layers": (128, 64),
        "activation":   "relu",
        "learning_rate_init": 0.001,
        "max_iter":     200,
    },
    {
        "run_name":     "MLP - Multi Layer (256, 128, 64)",
        "hidden_layers": (256, 128, 64),
        "activation":   "relu",
        "learning_rate_init": 0.001,
        "max_iter":     300,
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        mlp = MLPRegressor(
            hidden_layer_sizes=exp["hidden_layers"],
            activation=exp["activation"],
            solver="adam",
            learning_rate_init=exp["learning_rate_init"],
            max_iter=exp["max_iter"],
            early_stopping=True,        # stop when val loss stops improving
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        )

        mlp.fit(X_train_scaled, y_train)

        y_pred_train = mlp.predict(X_train_scaled)
        y_pred_test  = mlp.predict(X_test_scaled)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2_train   = r2_score(y_train, y_pred_train)

        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test    = r2_score(y_test, y_pred_test)

        mlflow.log_params({
            "hidden_layer_sizes":   str(exp["hidden_layers"]),
            "activation":           exp["activation"],
            "solver":               "adam",
            "learning_rate_init":   exp["learning_rate_init"],
            "max_iter":             exp["max_iter"],
            "early_stopping":       True,
            "n_iter_no_change":     15,
            "actual_iterations":    mlp.n_iter_,
            "n_layers":             mlp.n_layers_,
            "n_outputs":            mlp.n_outputs_,
            "sample_frac":          0.1,
            "split":                "time-based year<2021",
            "target":               "log1p(streams)",
        })
        mlflow.log_metrics({
            "RMSE":        rmse_test,
            "R2":          r2_test,
            "RMSE_train":  rmse_train,
            "R2_train":    r2_train,
        })

        print(f"\n{exp['run_name']}")
        print(f"  Camadas (n_layers_)  : {mlp.n_layers_}")
        print(f"  Outputs (n_outputs_) : {mlp.n_outputs_}")
        print(f"  Iterations           : {mlp.n_iter_}")
        print(f"  Train  RMSE={rmse_train:.4f}  R2={r2_train:.4f}")
        print(f"  Test   RMSE={rmse_test:.4f}  R2={r2_test:.4f}")

print("\nAll MLP experiments completed!")


# =============================================================================
# Experiment — MLPRegressor (Single Layer and Multi Layer)
#
# This experiment evaluated Multi-Layer Perceptron (MLP) neural networks for
# predicting log-transformed Spotify stream counts. The dataset was sampled
# down to 10% to keep training times manageable (MLP gradient descent is
# significantly slower than tree-based methods at this scale). A time-based
# split was used, training on data before 2021 and testing on 2021 onward.
#
# Preprocessing followed the same pipeline as previous experiments: target
# encoding for artist, genre, and region, and StandardScaler normalisation
# (essential for gradient-based optimisers like Adam).
#
# Four architectures were compared:
#   - Single layer with 64 neurons
#   - Single layer with 128 neurons
#   - Two hidden layers: 128 → 64
#   - Three hidden layers: 256 → 128 → 64
#
# All models used ReLU activation, the Adam optimiser, and early stopping
# (patience = 15 iterations with 10% validation split) to prevent overfitting.
#
# Deeper and wider networks are expected to capture more complex non-linear
# patterns than Linear Regression or Lasso, though they are less interpretable.
# The trade-off between model complexity, training time, and generalisation
# is the main focus of this analysis.
# =============================================================================

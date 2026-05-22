import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
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

# Sample to reduce training time
music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# Target Encoding
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


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print(" X_test_scaled  -> shape :", X_test_scaled.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Gradient Boosting Regressor")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {
        "run_name": "GB - fast  (n=50,  depth=3, lr=0.1)",
        "params": dict(n_estimators=50, learning_rate=0.1, max_depth=3,
                       subsample=0.8, min_samples_leaf=50, random_state=42)
    },
    {
        "run_name": "GB - medium (n=100, depth=4, lr=0.1)",
        "params": dict(n_estimators=100, learning_rate=0.1, max_depth=4,
                       subsample=0.8, min_samples_leaf=50, random_state=42)
    },
    {
        "run_name": "GB - deep  (n=200, depth=5, lr=0.05)",
        "params": dict(n_estimators=200, learning_rate=0.05, max_depth=5,
                       subsample=0.8, min_samples_leaf=50, random_state=42)
    },
]

for exp in experiments:

    with mlflow.start_run(run_name=exp["run_name"]):
        gb_reg = GradientBoostingRegressor(**exp["params"])
        gb_reg.fit(X_train_scaled, y_train)

        y_pred = gb_reg.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)


        mlflow.log_params({
            **exp["params"],
            "split": "time-based year<2021",
            "target": "log1p(streams)",
            "sample_frac": 0.2,
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

    print(f"  RMSE        : {rmse:.4f}")
    print(f"  R2          : {r2:.4f}")

print("\nAll GradientBoosting experiments completed!")



# music_regression.csv -> shape : (26173485, 17)
# After sampling -> shape : (5234697, 17)
#  X_train_scaled -> shape : (4093097, 16)
#  X_test_scaled  -> shape : (1141600, 16)
#
#
#   RMSE        : 0.6933
#   R2          : 0.7394
#
#   RMSE        : 0.6315
#   R2          : 0.7838
#
#   RMSE        : 0.6077
#   R2          : 0.7998

# This experiment used a Gradient Boosting Regressor to predict the target variable `log1p(streams)`.
# A random sample of 20% of the original dataset was used to reduce training time, and a time-based
# train-test split (`year < 2021`) was applied. Target encoding was performed for categorical variables
#     such as artist, genre, and region by calculating average log-transformed stream values.
#     The original categorical features were removed, and all input features were standardized before training.
#
# Three Gradient Boosting configurations were evaluated with different values for
# `n_estimators`, `max_depth`, and `learning_rate`. The experiments included a fast model,
# a medium-complexity model, and a deeper configuration with more trees and a lower learning rate.
#
# The best performance was achieved by the deep configuration with `n_estimators = 200`,
# `max_depth = 5`, and `learning_rate = 0.05`, which achieved an RMSE of 0.6077 and an
# R² score of 0.7998. The medium model reached an R² of 0.7838, while the baseline configuration
# achieved 0.7394. The results show that increasing model complexity improved predictive performance.
# Overall, Gradient Boosting demonstrated strong regression performance and effectively captured
# relationships in the streaming data.



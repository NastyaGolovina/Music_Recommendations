import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
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

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()


y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()


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
print("X_test_scaled -> shape :", X_test_scaled.shape)


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Regression")
mlflow.sklearn.autolog(log_models=False)

params = {
    "test_size": "year < 2021 / year >= 2021",
    "scaler": "StandardScaler",
    "target": "log1p(streams)",
    "region_encoding": "target_encoding",
}

with mlflow.start_run(run_name="LinearRegression - Baseline") as run:
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    y_pred = lr.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2",   r2)
    mlflow.log_params(params)

    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("Run completed!")

# ## Experiment 1 — Linear Regression Baseline
#
# This experiment established a baseline using a standard Linear Regression model trained on 26 million Spotify chart
# records. The dataset was split by time — training on data before 2021 and testing on 2021 and beyond — to realistically
# simulate predicting future stream counts. Categorical features such as artist, genre, and region were encoded using target
# encoding, and all features were scaled with StandardScaler as required by the model.
#
# The model achieved RMSE = 0.7870 and R2 = 0.6646, meaning it explains approximately 66% of the variation in
# stream counts. While this is a reasonable baseline, the model struggles because Linear Regression assumes all
# relationships between features and streams are strictly linear, which is not the case in reality. Stream popularity
# is driven by complex non-linear interactions — for example, being ranked #1 in the US is fundamentally different from
# being ranked #1 in a smaller market, and Linear Regression cannot capture this. Tree-based models such as Random Forest and
# XGBoost are expected to significantly outperform this result in subsequent experiments.
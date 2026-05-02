import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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

# # Sample to reduce training time
# music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
# print("After sampling -> shape :", music.shape)

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
mlflow.set_experiment("Spotify Streams - Stacking Regressor")
mlflow.sklearn.autolog(log_models=False)


experiments = [
    {
        "run_name": "Stacking - LR + KNN + RF -> LR",
        "estimators": [
            ("lr",  LinearRegression()),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
            ("rf",  RandomForestRegressor(n_estimators=100, random_state=42)),
        ],
        "final_estimator": LinearRegression(),
        "final_estimator_name": "LinearRegression",
    },
]
# experiments = [
#     {
#         "run_name": "Stacking - LR + KNN + RF -> LR",
#         "estimators": [
#             ("lr",  LinearRegression()),
#             ("knn", KNeighborsRegressor(n_neighbors=5)),
#             ("rf",  RandomForestRegressor(n_estimators=100, random_state=42)),
#         ],
#         "final_estimator": LinearRegression(),
#         "final_estimator_name": "LinearRegression",
#     },
#     {
#         "run_name": "Stacking - DT + KNN + GB -> LR",
#         "estimators": [
#             ("dt",  DecisionTreeRegressor(random_state=42)),
#             ("knn", KNeighborsRegressor(n_neighbors=5)),
#             ("rf",  RandomForestRegressor(n_estimators=100, random_state=42)),
#         ],
#         "final_estimator": LinearRegression(),
#         "final_estimator_name": "LinearRegression",
#     },
#     {
#         "run_name": "Stacking - LR + DT + RF -> RF",
#         "estimators": [
#             ("lr", LinearRegression()),
#             ("dt", DecisionTreeRegressor(random_state=42)),
#             ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
#         ],
#         "final_estimator": RandomForestRegressor(n_estimators=100, random_state=42),
#         "final_estimator_name": "RandomForestRegressor",
#     },
# ]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        stack_reg = StackingRegressor(
            estimators=exp["estimators"],
            final_estimator=exp["final_estimator"],
            n_jobs=-1
        )

        stack_reg.fit(X_train_scaled, y_train)
        y_pred = stack_reg.predict(X_test_scaled)

        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "base_estimators":   [name for name, _ in exp["estimators"]],
            "final_estimator":   exp["final_estimator_name"],
            "split":             "time-based year<2021",
            "target":            "log1p(streams)",
            "sample_frac":       0.2,
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")


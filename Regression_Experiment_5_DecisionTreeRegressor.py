import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
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

# Decision Tree does not require scaling
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - DecisionTreeRegressor")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "DecisionTree - max_depth=5",  "max_depth": 5},
    {"run_name": "DecisionTree - max_depth=10", "max_depth": 10},
    {"run_name": "DecisionTree - max_depth=15", "max_depth": 15},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        dt = DecisionTreeRegressor(
            max_depth=exp["max_depth"],
            min_samples_leaf=50,
            random_state=42
        )
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "max_depth":        exp["max_depth"],
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
dt_viz = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=50,
    random_state=42)
dt_viz.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(40, 10))
plot_tree(dt_viz, feature_names=X_train.columns.tolist(), filled=True, rounded=True, fontsize=8, ax=ax)
plt.title("Decision Tree Regressor (max_depth=5)")
plt.tight_layout()
plt.savefig("runs/png/DecisionTreeRegressor_tree.png", dpi=150)
plt.close()

with mlflow.start_run(run_name="DecisionTree - tree visualization"):
    mlflow.log_artifact("runs/png/DecisionTreeRegressor_tree.png")

print("Tree visualization saved.")


# ## Experiment 5 — Decision Tree Regressor
#
# This experiment applied a Decision Tree Regressor to predict log1p(streams).
# Decision Trees split the data based on feature thresholds to minimize prediction error.
# Unlike Linear Regression, Decision Trees can capture non-linear relationships.
# Three values of max_depth were tested (5, 10, 15) to find the optimal tree depth.
# min_samples_leaf=50 was set to avoid overfitting on small leaf nodes.
# Decision Trees do not require feature scaling unlike KNN or Linear Regression.
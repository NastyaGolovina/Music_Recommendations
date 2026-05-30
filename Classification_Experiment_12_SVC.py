import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

music = pd.read_csv("music_classification.csv", dtype={
    "rank":             "int16",
    "streams":          "float32",
    "is_hit":           "int8",
    "year":             "int16",
    "month":            "int8",
    "day":              "int8",
    "weekday":          "int8",
    "quarter":          "int8",
    "is_weekend":       "int8",
    "days_since_start": "int32",
    "artist_count":     "int32",
})

print("music_classification.csv -> shape :", music.shape)

music = music.sample(frac=0.002, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = train["is_hit"]
y_test  = test["is_hit"]

global_mean = y_train.mean()

# Artist → proportion of hits per artist
artist_mean = train.groupby("artist")["is_hit"].mean()
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# Genre → proportion of hits per genre
genre_mean = train.groupby("main_genre")["is_hit"].mean()
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# Region → proportion of hits per region
region_mean = train.groupby("region")["is_hit"].mean()
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test   -> shape :", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print("X_test_scaled   -> shape :", X_test_scaled.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - SVC Classifier")
mlflow.sklearn.autolog(log_models=False)

param_grid = [
    {"C": [0.1, 1.0], "kernel": ["rbf"], "gamma": ["scale"]},
]

grid = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
print(f"Best CV score   (GridSearchCV): {grid.best_score_:.4f}")

with mlflow.start_run(run_name="Experiment 12 - GridSearchCV SVC"):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type",     "SVC")
    mlflow.log_param("split",          "time-based year<2021")
    mlflow.log_param("sample_frac",    0.002)
    mlflow.log_param("C",              grid.best_params_["C"])
    mlflow.log_param("kernel",         grid.best_params_["kernel"])
    mlflow.log_param("gamma",          grid.best_params_["gamma"])
    mlflow.log_metric("Accuracy",      acc)
    mlflow.log_metric("Best_CV_score", grid.best_score_)
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

    print(f"\nExperiment 12 - GridSearchCV SVC")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - SVC Classifier"])
runs.to_csv("runs/Spotify_Streams_SVC_runs.csv", index=False)
print("CSV saved!")

# ## Experiment — SVC Classifier
#
# This experiment used a Support Vector Classifier (SVC) with an RBF kernel to
# classify songs as hits or non-hits. The dataset was split using a time-based
# approach (before 2021 for training and 2021 onwards for testing). Artist,
# genre, and region were target encoded, while numerical features were scaled
# using StandardScaler.
#
# Due to the computational cost of SVC, only 0.2% of the dataset was used.
# GridSearchCV (3-fold cross-validation) was applied to tune the hyperparameters.
# The best model achieved an accuracy of 0.9434 and a CV score of 0.9726.
#
# The model performed strongly on the majority class but showed lower recall for
# hit songs, indicating the impact of class imbalance. Overall, SVC produced
# competitive results and slightly outperformed Logistic Regression.
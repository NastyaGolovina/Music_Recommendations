import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
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

music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = train["is_hit"]
y_test  = test["is_hit"]

global_mean = y_train.mean()

# For each artist, calculate the proportion of their songs that are hits (0.0 to 1.0)
# Unknown artists in test set receive the global average (fallback)
artist_mean = train.groupby("artist")["is_hit"].mean()
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# For each genre, calculate the proportion of hits
genre_mean = train.groupby("main_genre")["is_hit"].mean()
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# For each region, calculate the proportion of hits
region_mean = train.groupby("region")["is_hit"].mean()
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test  -> shape :", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print("X_test_scaled   -> shape :", X_test_scaled.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Logistic Regression Classifier")
mlflow.sklearn.autolog(log_models=False)

param_grid = [
    {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"], "penalty": ["l2"]},
    {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["saga"],  "penalty": ["l1", "l2"]},
]

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train_scaled, y_train)

print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
print(f"Best CV score  (GridSearchCV): {grid.best_score_:.4f}")

with mlflow.start_run(run_name="Experiment 1 - GridSearchCV LogisticRegression"):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type",    "LogisticRegression")
    mlflow.log_param("split",         "time-based year<2021")
    mlflow.log_param("sample_frac",   0.05)
    mlflow.log_param("C",             grid.best_params_["C"])
    mlflow.log_param("solver",        grid.best_params_["solver"])
    mlflow.log_param("penalty",       grid.best_params_["penalty"])
    mlflow.log_metric("Accuracy",     acc)
    mlflow.log_metric("Best_CV_score", grid.best_score_)
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

    print(f"\nExperiment 1 - GridSearchCV LogisticRegression")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\nAll experiments completed!")


# ## Experiment 6 — Logistic Regression Classifier
#
# This experiment applied Logistic Regression to classify songs as hits or not hits.
# Logistic Regression estimates the probability of the positive class using a sigmoid
# function applied to a linear combination of input features. It is a fast, interpretable,
# and widely used linear baseline for binary classification problems.
# GridSearchCV with 5-fold cross-validation was used to find the best combination of
# regularisation strength (C), solver, and penalty type (L1 or L2).
# The best configuration was C=1.0, solver=saga, and L2 penalty, achieving a test
# accuracy of 0.9397 and a cross-validation score of 0.9548.
# While the model performs well on the majority class (not hit, F1=0.97), it struggles
# with the minority class (hit, F1=0.53) due to class imbalance in the dataset.
import pandas as pd
import numpy as np
import mlflow.sklearn
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


music = pd.read_csv("music_classification.csv", dtype={
    "rank": "int16",
    "streams": "float32",
    "is_hit": "int8",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "weekday": "int8",
    "quarter": "int8",
    "is_weekend": "int8",
    "days_since_start": "int32",
    "artist_count": "int32",
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
# Example: Drake → 0.85 means 85% of his songs are classified as hits
# Unknown artists in test set receive the global average (fallback)
artist_mean = train.groupby("artist")["is_hit"].mean()
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# For each genre, calculate the proportion of hits
# Example: pop → 0.61 means 61% of pop songs are hits
genre_mean = train.groupby("main_genre")["is_hit"].mean()
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# For each region, calculate the proportion of hits
# Example: United States → 0.74 means 74% of US chart songs are hits
region_mean = train.groupby("region")["is_hit"].mean()
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test -> shape :", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print("X_test_scaled -> shape :", X_test_scaled.shape)



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - KNN Classifier")
mlflow.sklearn.autolog(log_models=False)


def objective(trial):
    n_neighbors = trial.suggest_categorical("n_neighbors", [3, 5, 10, 20, 30, 50])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,

    )
    score = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring="accuracy", n_jobs=-1
    ).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print(f"\nBest parameters (Optuna): {study.best_params}")
print(f"Best CV score  (Optuna): {study.best_value:.4f}")

with mlflow.start_run(run_name="Experiment 1 - Optuna KNN"):
    best_params = study.best_params.copy()
    knn_optuna = KNeighborsClassifier(**best_params)
    knn_optuna.fit(X_train_scaled, y_train)
    y_pred_optuna = knn_optuna.predict(X_test_scaled)

    acc_optuna = accuracy_score(y_test, y_pred_optuna)

    mlflow.log_param("model_type",  "KNeighborsClassifier")
    mlflow.log_param("split",       "time-based year<2021")
    mlflow.log_param("sample_frac", 0.05)
    mlflow.log_param("n_neighbors", best_params["n_neighbors"])

    mlflow.log_metric("Accuracy",      acc_optuna)
    mlflow.log_metric("Best_CV_score", study.best_value)
    mlflow.log_text(classification_report(y_test, y_pred_optuna), "classification_report.txt")

    print(f"\nExperiment 1 - Optuna KNN")
    print(f"  Accuracy : {acc_optuna:.4f}")
    print(classification_report(y_test, y_pred_optuna))



print(f"  Optuna      Accuracy : {acc_optuna:.4f}")
print("\nAll experiments completed!")


#
# ### Data Shapes
# - Original dataset: (26,173,485, 18)
# - After sampling: (1,308,674, 18)
#
# - X_train: (1,023,318, 16)
# - X_test: (285,356, 16)
#
# - X_train_scaled: (1,023,318, 16)
# - X_test_scaled: (285,356, 16)

# ### Optuna Optimization (KNN)
# - Best parameters:
#   - n_neighbors = 5
#
# - Best CV score:
#   - 0.9669

# ### Experiment 1 — Optuna KNN Results
#
# - Accuracy: 0.9472
#
# Classification report:
#
# - Class 0
#   - Precision: 0.96
#   - Recall: 0.98
#   - F1-score: 0.97
#   - Support: 262,115
#
# - Class 1
#   - Precision: 0.74
#   - Recall: 0.54
#   - F1-score: 0.62
#   - Support: 23,241

# - Overall accuracy: 0.95
# - Macro avg:
#   - Precision: 0.85
#   - Recall: 0.76
#   - F1-score: 0.80
#
# - Weighted avg:
#   - Precision: 0.94
#   - Recall: 0.95
#   - F1-score: 0.94

# ### Final
# - Optuna KNN Accuracy: 0.9472



# This experiment used a K-Nearest Neighbors (KNN) classifier for binary classification to predict whether
# a song would become a hit (`is_hit`). KNN is a distance-based algorithm that classifies observations according
# to the labels of the nearest neighboring samples. Since KNN is sensitive to feature scales, all numerical
# variables were standardized using StandardScaler before training.
#
# To improve the quality of the input features, target encoding was applied to categorical variables such as
# artist, genre, and region. For each category, the average proportion of hit songs was calculated based on
# the training set and used as a new numerical feature. A time-based split (`year < 2021`) was used to separate
# training and test data, preserving the chronological order of observations. Additionally, only 5% of the
# original dataset was used to reduce computational costs.
#
# Hyperparameter tuning was performed using Optuna with 5-fold cross-validation. Several values of the
# `n_neighbors` parameter were tested, and the best configuration selected `n_neighbors = 5`. The model
# used the Minkowski distance metric with uniform neighbor weighting and the automatic algorithm selection mode.
#
# The optimized KNN model achieved an accuracy of 0.9472 on the test set and a cross-validation score of 0.9669.
# Training accuracy reached 0.9768, indicating strong predictive performance. However, the difference between
# training and test results may suggest a small degree of overfitting. Overall, the experiment showed that KNN,
# combined with feature engineering and hyperparameter optimization, produced strong classification results.

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


# music = music.sample(frac=0.05, random_state=42).reset_index(drop=True)
# print("After sampling -> shape :", music.shape)

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


# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - KNN Classifier")
mlflow.sklearn.autolog(log_models=False)

#
# with mlflow.start_run(run_name="Experiment 1 - Baseline KNN"):
#     pipeline = Pipeline([
#         ("scaler", StandardScaler()),
#         ("knn", KNeighborsClassifier(n_neighbors=5))
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#
#     acc = accuracy_score(y_test, y_pred)
#
#     mlflow.log_param("model_type", "KNeighborsClassifier")
#     mlflow.log_param("n_neighbors", 5)
#     mlflow.log_param("split", "time-based year<2021")
#     mlflow.log_param("sample_frac", 0.05)
#     mlflow.log_metric("Accuracy", acc)
#     mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
#
#     print(f"\nExperiment 1 - Baseline KNN")
#     print(f"  Accuracy : {acc:.4f}")
#     print(classification_report(y_test, y_pred))
#
#
# grid_pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("knn", KNeighborsClassifier())
# ])
#
# param_grid = [{
#     "knn__n_neighbors": [5, 10, 22, 50],
#     "knn__weights":     ["uniform"],
#     "knn__algorithm":   ["auto"]
# }]
#
# grid = GridSearchCV(
#     estimator=grid_pipeline,
#     param_grid=param_grid,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=-1,
#     verbose=1
# )
#
# grid.fit(X_train, y_train)
#
# print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
# print(f"Best CV score  (GridSearchCV): {grid.best_score_:.4f}")
#
# with mlflow.start_run(run_name="Experiment 2 - GridSearchCV KNN"):
#     best_grid_model = grid.best_estimator_
#     y_pred_grid = best_grid_model.predict(X_test)
#
#     acc_grid = accuracy_score(y_test, y_pred_grid)
#
#     mlflow.log_param("model_type",  "KNeighborsClassifier")
#     mlflow.log_param("split",       "time-based year<2021")
#     mlflow.log_param("sample_frac", 0.05)
#     mlflow.log_param("n_neighbors", grid.best_params_["knn__n_neighbors"])
#     mlflow.log_param("weights",     grid.best_params_["knn__weights"])
#     mlflow.log_param("algorithm",   grid.best_params_["knn__algorithm"])
#     mlflow.log_metric("Accuracy",   acc_grid)
#     mlflow.log_metric("Best_CV_score", grid.best_score_)
#     mlflow.log_text(classification_report(y_test, y_pred_grid), "classification_report.txt")
#
#     print(f"\nExperiment 2 - GridSearchCV KNN")
#     print(f"  Accuracy : {acc_grid:.4f}")
#     print(classification_report(y_test, y_pred_grid))


def objective(trial):
    n_neighbors = trial.suggest_categorical("n_neighbors", [3, 5, 7, 10, 15, 19, 22, 30, 50])
    weights     = trial.suggest_categorical("weights",     ["uniform", "distance"])
    algorithm   = trial.suggest_categorical("algorithm",   ["auto", "ball_tree", "kd_tree", "brute"])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm
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

with mlflow.start_run(run_name="Experiment 3 - Optuna KNN"):
    best_params = study.best_params.copy()
    knn_optuna = KNeighborsClassifier(**best_params)
    knn_optuna.fit(X_train_scaled, y_train)
    y_pred_optuna = knn_optuna.predict(X_test_scaled)

    acc_optuna = accuracy_score(y_test, y_pred_optuna)

    mlflow.log_param("model_type",  "KNeighborsClassifier")
    mlflow.log_param("split",       "time-based year<2021")
    mlflow.log_param("sample_frac", 0.05)
    mlflow.log_param("n_neighbors", best_params["n_neighbors"])
    mlflow.log_param("weights",     best_params["weights"])
    mlflow.log_param("algorithm",   best_params["algorithm"])
    mlflow.log_metric("Accuracy",      acc_optuna)
    mlflow.log_metric("Best_CV_score", study.best_value)
    mlflow.log_text(classification_report(y_test, y_pred_optuna), "classification_report.txt")

    print(f"\nExperiment 3 - Optuna KNN")
    print(f"  Accuracy : {acc_optuna:.4f}")
    print(classification_report(y_test, y_pred_optuna))


print("\n" + "="*50)
print("SUMMARY")
print("="*50)
# print(f"  Baseline    Accuracy : {acc:.4f}")
# print(f"  GridSearchCV Accuracy: {acc_grid:.4f}")
print(f"  Optuna      Accuracy : {acc_optuna:.4f}")
print("="*50)
print("\nAll experiments completed!")
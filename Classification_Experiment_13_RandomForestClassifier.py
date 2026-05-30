import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
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

music = music.sample(frac=0.02, random_state=42).reset_index(drop=True)
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

# Random Forest does not require scaling
drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test   -> shape :", X_test.shape)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - RandomForest Classifier")
mlflow.sklearn.autolog(log_models=False)

param_grid = {
    "n_estimators":     [100],
    "max_depth":        [10],
    "min_samples_leaf": [20],
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=4),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print(f"\nBest parameters (GridSearchCV): {grid.best_params_}")
print(f"Best CV score   (GridSearchCV): {grid.best_score_:.4f}")

with mlflow.start_run(run_name="Experiment 13 - GridSearchCV RandomForest"):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type",       "RandomForestClassifier")
    mlflow.log_param("split",            "time-based year<2021")
    mlflow.log_param("sample_frac",      0.02)
    mlflow.log_param("n_estimators",     grid.best_params_["n_estimators"])
    mlflow.log_param("max_depth",        grid.best_params_["max_depth"])
    mlflow.log_param("min_samples_leaf", grid.best_params_["min_samples_leaf"])
    mlflow.log_metric("Accuracy",        acc)
    mlflow.log_metric("Best_CV_score",   grid.best_score_)
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

    print(f"\nExperiment 13 - GridSearchCV RandomForest")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\nAll experiments completed!")

runs = mlflow.search_runs(experiment_names=["Spotify Streams - RandomForest Classifier"])
runs.to_csv("runs/Spotify_Streams_RandomForest_runs.csv", index=False)
print("CSV saved!")

# ## Experiment — Random Forest Classifier
#
# This experiment applied Random Forest to classify songs as hits or non-hits.
# A time-based split was used, with data before 2021 for training and data from
# 2021 onwards for testing. Artist, genre, and region were transformed using
# target encoding. Since Random Forest is tree-based, feature scaling was not required.
#
# GridSearchCV with 3-fold cross-validation was used to tune the model.
# The best configuration achieved an accuracy of 0.9514 and a CV score of 0.9698.
#
# The model performed strongly on the majority class and achieved better overall
# accuracy than Logistic Regression and SVC. Results suggest that Random Forest
# is able to capture non-linear relationships in the data more effectively than
# linear and kernel-based approaches.
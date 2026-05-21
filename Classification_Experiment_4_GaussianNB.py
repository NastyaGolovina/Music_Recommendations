import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
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

#music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
#print("After sampling -> shape :", music.shape)

train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = train["is_hit"]
y_test  = test["is_hit"]

global_mean = y_train.mean()

# target encoding — proportion of hits per artist/genre/region
artist_mean = train.groupby("artist")["is_hit"].mean()
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

genre_mean = train.groupby("main_genre")["is_hit"].mean()
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

region_mean = train.groupby("region")["is_hit"].mean()
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]
X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print(" X_test  -> shape :", X_test.shape)

# GaussianNB works better with scaled features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - GaussianNB")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "GaussianNB - baseline", "var_smoothing": 1e-9},
    {"run_name": "GaussianNB - smoothing=1e-6", "var_smoothing": 1e-6},
    {"run_name": "GaussianNB - smoothing=1e-3", "var_smoothing": 1e-3},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        gnb = GaussianNB(var_smoothing=exp["var_smoothing"])
        gnb.fit(X_train_scaled, y_train)

        y_train_pred = gnb.predict(X_train_scaled)
        y_test_pred  = gnb.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test,  y_test_pred)

        mlflow.log_param("model",          "GaussianNB")
        mlflow.log_param("var_smoothing",  exp["var_smoothing"])
        mlflow.log_param("split",          "time-based year<2021")
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy",  test_acc)
        mlflow.log_text(classification_report(y_test, y_test_pred), "classification_report.txt")

        print(f"\n{exp['run_name']}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_test_pred))

print("\nAll experiments completed!")

# ## Experiment 4 — Gaussian Naive Bayes Classifier
#
# This experiment applied Gaussian Naive Bayes to classify songs as hits or not hits.
# GaussianNB assumes each feature follows a Gaussian (normal) distribution and predicts
# the class with the highest probability using Bayes theorem.
# It is a fast and simple probabilistic model that works well as a baseline classifier.
# Three values of var_smoothing were tested to control the stability of the model.
# GaussianNB is particularly useful when the dataset is large since it trains very quickly.
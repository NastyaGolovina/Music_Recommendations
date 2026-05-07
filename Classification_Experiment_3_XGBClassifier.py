import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
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

# music = music.sample(frac=0.5, random_state=42).reset_index(drop=True)
# print("After sampling -> shape :", music.shape)


train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()


y_train = train["is_hit"]
y_test  = test["is_hit"]

global_mean = y_train.mean()




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
print("X_test -> shape :", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")




mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - XGBClassifier")

COMMON = {
    "random_state": 42,
    "eval_metric":  "logloss",
    "n_jobs":       4,
    "tree_method":  "hist",
}

experiments = [
    {
        "run_name": "XGB - baseline",
        "params": {
            **COMMON,
            "n_estimators":  100,
            "learning_rate": 0.1,
            "max_depth":     3,
        }
    },
    {
        "run_name": "XGB - deeper trees",
        "params": {
            **COMMON,
            "n_estimators":  200,
            "learning_rate": 0.1,
            "max_depth":     6,
        }
    },
    {
        "run_name": "XGB - low lr + more trees",
        "params": {
            **COMMON,
            "n_estimators":     300,
            "learning_rate":    0.05,
            "max_depth":        4,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
        }
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        print(f"\n {exp['run_name']}")

        xgb = XGBClassifier(**exp["params"])
        xgb.fit(X_train_scaled, y_train)

        y_train_pred = xgb.predict(X_train_scaled)
        y_test_pred  = xgb.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test,  y_test_pred)


        mlflow.log_param("model",            "XGBClassifier")
        mlflow.log_param("split",            "time-based year<2021")
        for k, v in exp["params"].items():
            mlflow.log_param(k, v)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy",  test_acc)


        report = classification_report(y_test, y_test_pred)
        mlflow.log_text(report, "classification_report.txt")

        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("\nClassification report:")
        print(report)

print("\nAll XGBoost experiments completed!")
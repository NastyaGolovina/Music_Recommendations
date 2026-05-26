import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import StandardScaler
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

music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

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

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train).astype("float32")
# X_test_scaled  = scaler.transform(X_test).astype("float32")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - AdaBoostClassifier")
mlflow.sklearn.autolog(log_models=False)

experiments = [
    {"run_name": "AdaBoost - n=50",  "n_estimators": 50,  "learning_rate": 0.1},
    {"run_name": "AdaBoost - n=100", "n_estimators": 100, "learning_rate": 0.1},
    {"run_name": "AdaBoost - lr=1.0", "n_estimators": 50, "learning_rate": 1.0},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=exp["n_estimators"],
            learning_rate=exp["learning_rate"],
            random_state=42
        )
        ada.fit(X_train, y_train)

        y_train_pred = ada.predict(X_train)
        y_test_pred  = ada.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test,  y_test_pred)

        mlflow.log_text(classification_report(y_test, y_test_pred), "classification_report.txt")

        print(f"\n{exp['run_name']}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_test_pred))

print("\nAll experiments completed!")

# ## Experiment 8 — AdaBoost Classifier
#
# This experiment applied AdaBoost Classifier to predict hit songs.
# AdaBoost works by sequentially training weak learners (shallow Decision Trees)
# where each new tree focuses more on the samples that previous trees misclassified.
# A DecisionTreeClassifier with max_depth=3 was used as the base estimator.
# Three configurations were tested varying n_estimators and learning_rate.
# AdaBoost is an ensemble method that typically outperforms a single Decision Tree
# by combining multiple weak learners into a strong classifier.
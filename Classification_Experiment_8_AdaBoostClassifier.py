import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

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

        train_acc      = accuracy_score(y_train, y_train_pred)
        test_acc       = accuracy_score(y_test,  y_test_pred)
        test_f1        = f1_score(y_test, y_test_pred, zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall    = recall_score(y_test, y_test_pred, zero_division=0)

        mlflow.log_metric("Accuracy",  test_acc)
        mlflow.log_metric("F1",        test_f1)
        mlflow.log_metric("Precision", test_precision)
        mlflow.log_metric("Recall",    test_recall)
        mlflow.log_metric("train_acc", train_acc)
        mlflow.log_text(classification_report(y_test, y_test_pred), "classification_report.txt")

        print(f"\n{exp['run_name']}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print(f"  F1             : {test_f1:.4f}")
        print(f"  Precision      : {test_precision:.4f}")
        print(f"  Recall         : {test_recall:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_test_pred))

print("\nAll experiments completed!")

# =============================================================================
# Experiment 8 — AdaBoost Classifier
# =============================================================================
#
# Goal:
#   Classify songs as hits (is_hit=1) or not (is_hit=0) using AdaBoost.
#   AdaBoost trains weak learners (shallow Decision Trees) sequentially — each
#   new tree focuses more on the samples previously misclassified, combining
#   them into a strong classifier. A DecisionTreeClassifier (max_depth=3,
#   Gini criterion) is used as the base estimator.
#
# Setup:
#   - 20% sample of music_classification.csv (time-based split: train < 2021, test >= 2021)
#   - Target encoding applied to artist, genre, region
#   - No scaling needed (tree-based base estimator)
#   - Three configurations tested varying n_estimators and learning_rate
#
# Results:
#   Run                      Accuracy   F1      Precision  Recall   Train Acc  Duration
#   n=50,  lr=0.1            0.9443     0.6005  0.7139     0.5182   0.9615     5.3min
#   n=100, lr=0.1            0.9457     0.6311  0.6989     0.5753   0.9623     9.7min
#   n=50,  lr=1.0            0.9480     0.6819  0.6737     0.6903   0.9706     4.8min
#
# Analysis:
#   The best configuration is n=50 with learning_rate=1.0, achieving the highest
#   F1 (0.6819) and recall (0.6903) while also being faster than n=100 (4.8min vs 9.7min).
#   A higher learning rate makes each tree contribute more aggressively, which
#   improves recall on the minority hit class. Compared to GaussianNB (F1=0.5677)
#   and Perceptron (F1=0.1106), AdaBoost is significantly stronger. However the
#   gap between training accuracy (0.9706) and test accuracy (0.9480) indicates
#   some overfitting, especially at lr=1.0.
# =============================================================================
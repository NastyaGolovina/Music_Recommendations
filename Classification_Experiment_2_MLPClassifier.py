import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
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

music = music.sample(frac=0.1, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)


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
mlflow.set_experiment("Spotify Streams - MLPClassifier")
mlflow.sklearn.autolog(log_models=False)
experiments = [
    {
        "run_name": "MLP - baseline",
        "params": {
            "hidden_layer_sizes": (100,),
            "activation":         "relu",
            "max_iter":           50  ,
            "random_state":       42,
        }
    },
    {
        "run_name": "MLP - deeper network",
        "params": {
            "hidden_layer_sizes": (100, 50),
            "activation":         "relu",
            "max_iter":           50  ,
            "random_state":       42,
        }
    },
    {
        "run_name": "MLP - tanh activation",
        "params": {
            "hidden_layer_sizes": (100,),
            "activation":         "tanh",
            "max_iter":           50  ,
            "random_state":       42,
        }
    },
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        mlp = MLPClassifier(**exp["params"])
        mlp.fit(X_train_scaled, y_train)

        y_train_pred = mlp.predict(X_train_scaled)
        y_test_pred  = mlp.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc  = accuracy_score(y_test,  y_test_pred)

        mlflow.log_param("model",               "MLPClassifier")
        mlflow.log_param("split",               "time-based year<2021")
        mlflow.log_param("hidden_layer_sizes",  str(exp["params"]["hidden_layer_sizes"]))
        mlflow.log_param("activation",          exp["params"]["activation"])
        mlflow.log_param("max_iter",            exp["params"]["max_iter"])
        mlflow.log_param("random_state",        exp["params"]["random_state"])
        mlflow.log_metric("train_accuracy",     train_acc)
        mlflow.log_metric("test_accuracy",      test_acc)
        mlflow.log_text(classification_report(y_test, y_test_pred), "classification_report.txt")

        print(f"\n{exp['run_name']}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_test_pred))

print("\nAll experiments completed!")



# This experiment used a Multi-Layer Perceptron (MLP) classifier to predict whether a song would become
# a hit (`is_hit`). A time-based train-test split (`year < 2021`) was applied, and 10% of the original
# dataset was used for training. Target encoding was applied to categorical variables such as artist, genre,
# and region, and all features were standardized using StandardScaler before model training.
#
# Three MLP configurations were tested: a baseline model with one hidden layer `(100)`, a deeper network
# `(100, 50)`, and a model with Tanh activation. All models used the Adam optimizer and were trained for 50
# iterations.
#
# The best results were achieved by the deeper network model `(100, 50)`, which reached a test accuracy of
# 0.9567 and a training accuracy of 0.9783. The results indicate strong predictive performance, with only
#     a small difference between training and test scores, suggesting mild overfitting. Overall, the MLP
#     classifier performed well and effectively captured complex patterns in the data.


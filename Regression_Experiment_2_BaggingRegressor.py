import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

music = pd.read_csv("music_regression.csv", dtype={
    "rank": "int16",
    "streams": "float32",
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "weekday": "int8",
    "quarter": "int8",
    "is_weekend": "int8",
    "days_since_start": "int32",
    "artist_count": "int32",
})

print("music_regression.csv -> shape :", music.shape)

# Sample to reduce training time
music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)


train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()


y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# For each artist, calculate the average log1p(streams)
# log1p compresses large stream counts to avoid outliers dominating the mean
# Example: Drake → 13.2, unknown artist → global_mean (fallback)
artist_mean = train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# For each genre, calculate the average log1p(streams)
# Captures how streamable each genre is on average
genre_mean = train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# For each region, calculate the average log1p(streams)
# Replaces 70 one-hot columns with a single numeric column
region_mean = train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)


drop_cols = ["streams", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

print(" X_train -> shape :", X_train.shape)
print("X_test -> shape :", X_test.shape)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train).astype("float32")
# X_test_scaled  = scaler.transform(X_test).astype("float32")
#
# print(" X_train_scaled -> shape :", X_train_scaled.shape)
# print("X_test_scaled -> shape :", X_test_scaled.shape)





mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Bagging Regressor")
mlflow.sklearn.autolog(log_models=False)


experiments = [
    # {"run_name": "Bagging - n_estimators=10",  "n_estimators": 10,  "max_samples": 0.8},
     {"run_name": "Bagging - n_estimators=50",  "n_estimators": 50,  "max_samples": 0.8},
    #{"run_name": "Bagging - n_estimators=100", "n_estimators": 100, "max_samples": 0.6},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        bag_reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, random_state=42),
            n_estimators=exp["n_estimators"],
            max_samples=exp["max_samples"],
            bootstrap=True,
            random_state=42,
            n_jobs=4
        )

        bag_reg.fit(X_train, y_train)
        y_pred = bag_reg.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        mlflow.log_params({
            "n_estimators": exp["n_estimators"],
            "max_samples":  exp["max_samples"],
            "bootstrap":    True,
            "split":        "time-based year<2021",
            "target":       "log1p(streams)",
        })
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2",   r2)

        print(f"\n{exp['run_name']}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")

print("\nAll experiments completed!")



# The experiment evaluated a Bagging Regressor for predicting Spotify track stream counts on a dataset of
# approximately 26 million records. The target variable was log-transformed using log1p to reduce the
# impact of extreme outliers. The dataset was sampled down to 20 percent and split temporally, with records
#     before 2021 used for training and records from 2021 onward used for testing, to reflect a realistic
#     forecasting scenario.
#
# Feature engineering relied on target encoding for three high-cardinality categorical columns: artist, genre,
# and region. Each was replaced with the mean log-transformed stream count computed on the training set, with
# unseen categories falling back to the global mean. Temporal features such as day, month, weekday, and quarter,
# along with rank and artist count, were kept as numeric inputs. StandardScaler was not applied since tree-based
# models do not require feature scaling.
#
# The ensemble consisted of 50 Decision Tree base learners, each trained on a bootstrap sample covering 80 percent
# of the training data. Individual trees were constrained to a maximum depth of 10 and a minimum of 50 samples
# per leaf to limit overfitting at the base estimator level. Training ran across 4 parallel jobs and completed
# in approximately 5.5 minutes.
#
# The model achieved a test R2 of 0.796 and a test RMSE of 0.6135 on the log scale. Training R2 was 0.861,
# giving a gap of roughly 0.065 between train and test performance. This suggests moderate overfitting, likely
# because the ensemble picks up artist and region specific patterns from the training period that do not fully
# carry over to post-2021 data. Overall the result is a solid baseline and closely matches the Stacking experiment
# run under similar constraints, indicating that the shared tree configuration is the main factor limiting
# performance rather than the choice of ensemble method.





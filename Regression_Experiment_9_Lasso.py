import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# Load dataset
# =============================================================================
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

# Sample 20% to reduce training time (same approach as BaggingRegressor/StackingRegressor)
music = music.sample(frac=0.2, random_state=42).reset_index(drop=True)
print("After sampling -> shape :", music.shape)

# =============================================================================
# Time-based train/test split
# =============================================================================
train = music[music["year"] < 2021].copy()
test  = music[music["year"] >= 2021].copy()

y_train = np.log1p(train["streams"])
y_test  = np.log1p(test["streams"])

global_mean = y_train.mean()

# =============================================================================
# Target Encoding for high-cardinality categorical columns
# =============================================================================
# Artist: replace with mean log1p(streams) per artist on training data
artist_mean = train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
test["artist_te"]  = test["artist"].map(artist_mean).fillna(global_mean)

# Genre: replace with mean log1p(streams) per genre
genre_mean = train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
train["genre_te"] = train["main_genre"].map(genre_mean).fillna(global_mean)
test["genre_te"]  = test["main_genre"].map(genre_mean).fillna(global_mean)

# Region: replaces ~70 one-hot columns with a single numeric column
region_mean = train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())
train["region_te"] = train["region"].map(region_mean).fillna(global_mean)
test["region_te"]  = test["region"].map(region_mean).fillna(global_mean)

# =============================================================================
# Feature matrix
# =============================================================================
drop_cols = ["streams", "artist", "main_genre", "region"]

X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

# Lasso is a linear model — scaling is required
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train).astype("float32")
X_test_scaled  = scaler.transform(X_test).astype("float32")

print(" X_train_scaled -> shape :", X_train_scaled.shape)
print(" X_test_scaled  -> shape :", X_test_scaled.shape)

# =============================================================================
# MLflow — Lasso with Alpha Analysis
# =============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spotify Streams - Regression")
mlflow.sklearn.autolog(log_models=False)

# Alpha controls the strength of L1 regularisation:
#   alpha → 0   : approaches OLS (like Linear Regression)
#   alpha → ∞   : all coefficients shrink to zero
# We test a range to identify the optimal balance between bias and variance.
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

best_rmse  = float("inf")
best_alpha = None

for alpha in alphas:
    run_name = f"Lasso - alpha={alpha}"
    with mlflow.start_run(run_name=run_name):

        lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        lasso.fit(X_train_scaled, y_train)

        y_pred_train = lasso.predict(X_train_scaled)
        y_pred_test  = lasso.predict(X_test_scaled)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2_train   = r2_score(y_train, y_pred_train)

        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test    = r2_score(y_test, y_pred_test)

        # Count features driven to zero by L1 (Lasso's sparsity property)
        n_zero_coefs = int(np.sum(np.abs(lasso.coef_) < 1e-6))
        n_features   = X_train_scaled.shape[1]

        mlflow.log_params({
            "alpha":            alpha,
            "max_iter":         10000,
            "sample_frac":      0.2,
            "test_size":        "year < 2021 / year >= 2021",
            "scaler":           "StandardScaler",
            "target":           "log1p(streams)",
            "region_encoding":  "target_encoding",
        })
        mlflow.log_metrics({
            "RMSE":            rmse_test,
            "R2":              r2_test,
            "RMSE_train":      rmse_train,
            "R2_train":        r2_train,
            "zero_coefs":      n_zero_coefs,
            "nonzero_coefs":   n_features - n_zero_coefs,
        })

        print(f"\nalpha={alpha}")
        print(f"  Train  RMSE={rmse_train:.4f}  R2={r2_train:.4f}")
        print(f"  Test   RMSE={rmse_test:.4f}  R2={r2_test:.4f}")
        print(f"  Zeroed coefficients: {n_zero_coefs}/{n_features}")

        if rmse_test < best_rmse:
            best_rmse  = rmse_test
            best_alpha = alpha

print(f"\nBest alpha: {best_alpha}  (Test RMSE = {best_rmse:.4f})")
print("All Lasso experiments completed!")


# =============================================================================
# Experiment — Lasso Regression with Alpha Analysis
#
# This experiment evaluated Lasso (L1-regularised Linear Regression) for
# predicting Spotify stream counts. The target variable was log-transformed
# using log1p to reduce the impact of outliers. The dataset was sampled to
# 20% (same as BaggingRegressor and StackingRegressor experiments) and split
# temporally, with training data from before 2021 and test data from 2021
# onward to simulate a realistic forecasting scenario.
#
# Categorical features (artist, genre, region) were replaced via target
# encoding, and all features were standardised with StandardScaler since
# Lasso is sensitive to feature scale.
#
# Six alpha values were evaluated: 0.0001, 0.001, 0.01, 0.1, 1.0, and 10.0.
# Alpha controls the L1 penalty strength — small values leave the model close
# to ordinary least squares, while large values push coefficients toward zero,
# effectively performing automatic feature selection.
#
# The key advantage of Lasso over plain Linear Regression is sparsity: at
# higher alphas, some coefficients are driven exactly to zero, revealing
# which features are least informative for predicting streams. This makes
# the model more interpretable and can improve generalisation when irrelevant
# features are present.
#
# Lasso shares the fundamental limitation of all linear models — it cannot
# capture non-linear interactions between features (e.g. rank x region
# interactions). Tree-based ensemble methods are expected to outperform it,
# but Lasso provides a stronger and more principled baseline than plain
# Linear Regression.
# =============================================================================

# =============================================================================
# Experimento — Regressão Lasso com Análise de Alpha
#
# Este experimento avaliou o Lasso (Regressão Linear com regularização L1)
# para prever o número de streams de músicas no Spotify. A variável alvo foi
# transformada com log1p para reduzir o impacto de valores extremos. O dataset
# foi amostrado a 20% (igual aos experimentos BaggingRegressor e
# StackingRegressor) e dividido temporalmente, com dados de treino anteriores
# a 2021 e dados de teste a partir de 2021, para simular um cenário realista
# de previsão.
#
# As features categóricas (artista, género, região) foram substituídas por
# target encoding, e todas as features foram normalizadas com StandardScaler,
# uma vez que o Lasso é sensível à escala dos dados.
#
# Foram avaliados seis valores de alpha: 0.0001, 0.001, 0.01, 0.1, 1.0 e 10.0.
# O alpha controla a força da penalização L1 — valores pequenos deixam o modelo
# próximo da regressão linear simples, enquanto valores grandes empurram os
# coeficientes para zero, realizando seleção automática de features.
#
# A principal vantagem do Lasso sobre a Regressão Linear simples é a
# sparsidade: com alphas maiores, alguns coeficientes são reduzidos exatamente
# a zero, revelando quais features são menos informativas para prever streams.
# Isto torna o modelo mais interpretável e pode melhorar a generalização quando
# existem features irrelevantes.
#
# O Lasso partilha a limitação fundamental de todos os modelos lineares — não
# consegue capturar interações não-lineares entre features (por exemplo,
# interações entre rank e região). Os métodos de ensemble baseados em árvores
# deverão superar o Lasso, mas este fornece uma baseline mais robusta e
# fundamentada do que a Regressão Linear simples.
# =============================================================================

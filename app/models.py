"""
models.py
---------
Implements four forecasting models:
  1. SARIMA  (statsmodels)
  2. Prophet (Meta / Facebook)
  3. XGBoost (gradient-boosted trees with lag features)
  4. LSTM    (TensorFlow / Keras)

Each model class exposes:
  .fit(train_df)  – train on a weekly time-series DataFrame (cols: ds, y)
  .predict(n)     – return n-step-ahead forecasts as a list of floats
  .evaluate(test_df) – return {"rmse": …, "mae": …, "mape": …}
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# SHARED METRIC HELPER
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    # MAPE – guard against zeros
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# 1. SARIMA
# ──────────────────────────────────────────────────────────────────────────────

class SARIMAModel:
    """Seasonal ARIMA via statsmodels auto-order selection (fast defaults)."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.model_fit      = None
        self.history        = None

    def fit(self, train_df: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = train_df["y"].values
        self.history = list(y)
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False, maxiter=200)
        return self

    def predict(self, n: int = 8) -> list:
        """Forecast n steps ahead."""
        forecast = self.model_fit.forecast(steps=n)
        return [max(0.0, float(v)) for v in forecast]

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        preds = self.predict(len(test_df))
        return compute_metrics(test_df["y"].values, preds)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# 2. PROPHET
# ──────────────────────────────────────────────────────────────────────────────

class ProphetModel:
    """Facebook / Meta Prophet with yearly & weekly seasonality."""

    def __init__(self):
        self.model = None

    def fit(self, train_df: pd.DataFrame):
        from prophet import Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        self.model.fit(train_df[["ds", "y"]])
        return self

    def predict(self, n: int = 8, last_date: pd.Timestamp = None) -> list:
        future = self.model.make_future_dataframe(periods=n, freq="W")
        forecast = self.model.predict(future)
        return [max(0.0, float(v)) for v in forecast["yhat"].values[-n:]]

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        future    = self.model.make_future_dataframe(periods=len(test_df), freq="W")
        forecast  = self.model.predict(future)
        preds     = forecast["yhat"].values[-len(test_df):]
        return compute_metrics(test_df["y"].values, preds)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# 3. XGBOOST
# ──────────────────────────────────────────────────────────────────────────────

from app.preprocessing import FEATURE_COLS, add_features

class XGBoostModel:
    """XGBoost regressor trained on engineered lag + calendar features."""

    def __init__(self):
        self.model   = None
        self.scaler  = MinMaxScaler()
        self._train  = None   # keep training set for recursive forecasting

    def fit(self, train_df: pd.DataFrame):
        from xgboost import XGBRegressor
        df = add_features(train_df)
        self._train = df.copy()
        X = df[FEATURE_COLS].values
        y = df["y"].values
        X_scaled = self.scaler.fit_transform(X)
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)
        return self

    def _build_row(self, history: pd.DataFrame) -> np.ndarray:
        """Build a single feature row from the most recent history."""
        df = add_features(history)
        if len(df) == 0:
            return None
        row = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        return self.scaler.transform(row)

    def predict(self, n: int = 8) -> list:
        """Recursive multi-step forecast."""
        history = self._train.copy()
        preds   = []
        for _ in range(n):
            X_row = self._build_row(history)
            if X_row is None:
                preds.append(0.0)
                continue
            yhat = float(self.model.predict(X_row)[0])
            yhat = max(0.0, yhat)
            preds.append(yhat)
            # Append the prediction as the next observation
            last_date = history["ds"].iloc[-1]
            new_row   = pd.DataFrame({
                "ds": [last_date + pd.Timedelta(weeks=1)],
                "y":  [yhat],
            })
            history = pd.concat([history, new_row], ignore_index=True)
        return preds

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        preds = self.predict(len(test_df))
        return compute_metrics(test_df["y"].values, preds)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. LSTM
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel:
    """
    Single-layer LSTM (TensorFlow / Keras) trained on sliding windows of the
    normalised sales series. Uses a look-back of 13 weeks.
    """

    LOOK_BACK = 13

    def __init__(self, epochs: int = 50, batch_size: int = 16):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model      = None
        self.scaler     = MinMaxScaler(feature_range=(0, 1))
        self._last_seq  = None   # last sequence used for forecasting

    # ── helpers ──

    def _make_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.LOOK_BACK, len(data)):
            X.append(data[i - self.LOOK_BACK:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        values = train_df["y"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        X, y = self._make_sequences(scaled)
        X    = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.LOOK_BACK, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=0,
        )

        # Save the last LOOK_BACK observations for future forecasting
        self._last_seq = scaled[-self.LOOK_BACK:]
        return self

    def predict(self, n: int = 8) -> list:
        seq   = self._last_seq.copy()
        preds = []
        for _ in range(n):
            X    = seq.reshape(1, self.LOOK_BACK, 1)
            yhat = float(self.model.predict(X, verbose=0)[0, 0])
            preds.append(yhat)
            seq  = np.append(seq[1:], [[yhat]], axis=0)

        # Inverse-scale
        preds_inv = self.scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()
        return [max(0.0, float(v)) for v in preds_inv]

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        preds = self.predict(len(test_df))
        return compute_metrics(test_df["y"].values, preds)

    def save(self, path: str):
        """Save LSTM weights separately; store the rest with joblib."""
        import os
        weights_path = path.replace(".pkl", "_weights.h5")
        self.model.save_weights(weights_path)
        _model_backup  = self.model
        self.model     = None
        joblib.dump(self, path)
        self.model     = _model_backup

    def load_weights(self, path: str):
        """Reload weights after joblib.load."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        # Rebuild architecture then load weights
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.LOOK_BACK, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        weights_path = path.replace(".pkl", "_weights.h5")
        self.model.load_weights(weights_path)

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        obj.load_weights(path)
        return obj

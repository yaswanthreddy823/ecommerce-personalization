"""Category demand forecasting: ARIMA + optional LSTM.

Production used LSTM and ARIMA time series models for category-level
demand prediction. This demo generates a year of daily demand with trend
and weekly seasonality, fits an ARIMA model, and compares it against a
naive last-value forecast on a 30-day holdout. An LSTM variant (PyTorch)
runs when torch is installed and is skipped gracefully otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def make_series(n_days: int = 365, seed: int = 42) -> pd.Series:
    """Daily demand: upward trend + weekly seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    trend = 100 + 0.3 * t
    weekly = 25 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 15, size=n_days)
    values = np.maximum(trend + weekly + noise, 10)
    index = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.Series(values, index=index, name="demand")


def arima_forecast(train: pd.Series, steps: int = 30) -> np.ndarray:
    # Seasonal ARIMA: weekly seasonality (period 7) matches the demand
    # pattern; a non-seasonal ARIMA degenerates to the naive forecast here.
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(steps=steps))


def lstm_forecast(train: pd.Series, steps: int = 30,
                  window: int = 30, epochs: int = 20) -> np.ndarray | None:
    """Small PyTorch LSTM forecaster. Returns None when torch is missing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch not installed; skipping LSTM variant.")
        return None
    values = train.to_numpy(dtype=np.float32)
    mean, std = values.mean(), values.std()
    scaled = (values - mean) / std
    X = np.stack([scaled[i:i + window] for i in range(len(scaled) - window)])
    y = scaled[window:]
    X_t = torch.tensor(X).unsqueeze(-1)
    y_t = torch.tensor(y)

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    torch.manual_seed(42)
    model = Forecaster()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    history = list(scaled[-window:])
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(history[-window:]).unsqueeze(0).unsqueeze(-1)
            pred = float(model(x).item())
            preds.append(pred)
            history.append(pred)
    return np.asarray(preds) * std + mean


def main() -> None:
    series = make_series()
    train, test = series.iloc[:-30], series.iloc[-30:]
    naive = np.full(30, train.iloc[-1])
    arima = arima_forecast(train)
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    print(f"30-day holdout MAE: naive {mae(naive, test.to_numpy()):.1f}, "
          f"ARIMA {mae(arima, test.to_numpy()):.1f}")
    lstm = lstm_forecast(train)
    if lstm is not None:
        print(f"30-day holdout MAE: LSTM {mae(lstm, test.to_numpy()):.1f}")


if __name__ == "__main__":
    main()

"""Anomaly detection for e-commerce payment fraud (demonstration).

Production monitored transactions in real time with anomaly detection
to flag high-risk payments. This demo trains an Isolation Forest on
synthetic payment features and reports how well flagged transactions
match the synthetic ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def make_payments(n: int = 10000, fraud_rate: float = 0.02,
                  seed: int = 42) -> pd.DataFrame:
    """Synthetic payments: legitimate profile vs fraud profile."""
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    fraud = is_fraud == 1
    amount = rng.lognormal(mean=3.5, sigma=0.9, size=n)
    hour = rng.integers(0, 24, size=n)
    velocity = rng.poisson(lam=1.0, size=n)
    new_device = (rng.random(n) < 0.15).astype(int)
    m = int(fraud.sum())
    amount[fraud] = rng.lognormal(mean=4.6, sigma=1.1, size=m)
    hour[fraud] = rng.integers(0, 6, size=m)
    velocity[fraud] = rng.poisson(lam=4.0, size=m)
    new_device[fraud] = (rng.random(m) < 0.5).astype(int)
    return pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour": hour,
            "velocity": velocity,
            "new_device": new_device,
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    df = make_payments()
    X = df[["amount", "hour", "velocity", "new_device"]]
    model = IsolationForest(n_estimators=200, contamination=0.02,
                            random_state=42, n_jobs=-1)
    flagged = (model.fit_predict(X) == -1).astype(int)
    print(f"Flagged {flagged.sum()} of {len(df)} payments "
          f"({flagged.mean():.1%}).")
    print(classification_report(df["is_fraud"], flagged, digits=3,
                                target_names=["legit", "flagged"]))


if __name__ == "__main__":
    main()

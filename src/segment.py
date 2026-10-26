"""Customer segmentation with KMeans on behavioral features.

Production segmented customers in real time from behavioral features to
power targeted campaigns. This demo clusters users on spend, order
frequency, average order value, category breadth, and clicks, then
profiles each cluster in plain language.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from etl import build_user_features, clean_orders, make_data


def segment(features: pd.DataFrame, n_clusters: int = 4,
            seed: int = 42) -> pd.DataFrame:
    """Assign a segment label per user and return cluster profiles."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=seed).fit_predict(scaled)
    features = features.copy()
    features["segment"] = labels
    profiles = features.groupby("segment").mean(numeric_only=True)
    profiles["users"] = features.groupby("segment").size()
    return features, profiles


def describe_profile(row: pd.Series) -> str:
    """Plain-language label from a cluster's average behavior."""
    if row["clicks"] > 60 and row["orders"] < 8:
        return "browsers who rarely buy"
    if row["total_spend"] > 800:
        return "high-value buyers"
    if row["total_spend"] > 350:
        return "steady mid-value buyers"
    return "low-spend occasional buyers"


def main() -> None:
    data = make_data()
    orders = clean_orders(data["orders"])
    features = build_user_features(orders, data["clicks"])
    labeled, profiles = segment(features)
    print(f"Segmented {len(labeled):,} users into {len(profiles)} clusters:")
    for seg, row in profiles.iterrows():
        print(f"  segment {seg} ({describe_profile(row)}): "
              f"{int(row['users'])} users, mean spend "
              f"${row['total_spend']:.0f}, {row['orders']:.1f} orders")


if __name__ == "__main__":
    main()

"""Smoke tests for the e-commerce platform modules.

Run:  python -m pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly import make_payments
from etl import build_user_features, build_user_item, clean_orders, make_data
from forecast import arima_forecast, make_series
from recommend import content_scores, item_cf_scores, recommend
from segment import segment


def test_etl_shapes():
    data = make_data(n_users=500, n_products=100, n_orders=5000)
    orders = clean_orders(data["orders"])
    assert len(orders) > 0
    assert orders["price"].min() > 0
    user_item = build_user_item(orders)
    assert user_item.shape[0] > 0 and user_item.shape[1] > 0
    features = build_user_features(orders, data["clicks"])
    assert len(features) == user_item.shape[0]


def test_recommend_returns_unpurchased_items():
    data = make_data(n_users=500, n_products=100, n_orders=5000)
    orders = clean_orders(data["orders"])
    user_item = build_user_item(orders)
    cf = item_cf_scores(user_item)
    content = content_scores(data["products"])
    user_id = int(user_item.index[0])
    purchased = set(user_item.columns[user_item.loc[user_id].to_numpy() > 0])
    recs = recommend(user_id, user_item, cf, content, k=5)
    assert len(recs) == 5
    assert not any(p in purchased for p, _ in recs)


def test_forecast_length_and_finite():
    series = make_series(n_days=200)
    forecast = arima_forecast(series.iloc[:-30], steps=30)
    assert len(forecast) == 30
    assert np.all(np.isfinite(forecast))


def test_segmentation_cluster_count():
    data = make_data(n_users=500, n_products=100, n_orders=5000)
    orders = clean_orders(data["orders"])
    features = build_user_features(orders, data["clicks"])
    labeled, profiles = segment(features, n_clusters=4)
    assert labeled["segment"].nunique() == 4
    assert profiles["users"].sum() == len(features)


def test_anomaly_flags_minority():
    df = make_payments(n=5000)
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=100, contamination=0.02,
                            random_state=42)
    flagged = (model.fit_predict(
        df[["amount", "hour", "velocity", "new_device"]]) == -1)
    assert 0.005 < flagged.mean() < 0.05

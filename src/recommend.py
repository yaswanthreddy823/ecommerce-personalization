"""Hybrid recommendations: collaborative + content-based filtering.

- Item-item collaborative filtering: products are similar when the same
  users buy them (cosine similarity over user purchase vectors).
- Content-based filtering: products are similar when their descriptions
  match (TF-IDF cosine similarity).
- Hybrid: combine both signals, exclude already-purchased items, and
  return the top-k.

Evaluation holds out each user's most recent orders and measures
precision@k -- the share of recommended items the user actually bought.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from etl import build_user_item, clean_orders, make_data


def item_cf_scores(user_item: pd.DataFrame) -> np.ndarray:
    """Item-item cosine similarity over user purchase vectors."""
    matrix = user_item.to_numpy(dtype=float)
    return cosine_similarity(matrix.T)


def content_scores(products: pd.DataFrame) -> np.ndarray:
    """Item-item cosine similarity over TF-IDF descriptions."""
    text = products["category"] + " " + products["description"]
    tfidf = TfidfVectorizer().fit_transform(text)
    return cosine_similarity(tfidf)


def recommend(user_id: int, user_item: pd.DataFrame,
              cf: np.ndarray, content: np.ndarray,
              k: int = 5, cf_weight: float = 0.6) -> list[tuple[int, float]]:
    """Top-k product recommendations for a user (product_id, score)."""
    if user_id not in user_item.index:
        return []
    purchased = set(user_item.columns[user_item.loc[user_id].to_numpy() > 0])
    user_vector = user_item.loc[user_id].to_numpy(dtype=float)
    # Score each candidate by similarity to items the user already bought.
    cf_score = user_vector @ cf
    content_score = user_vector @ content
    combined = cf_weight * cf_score + (1 - cf_weight) * content_score
    order = np.argsort(combined)[::-1]
    results = []
    for idx in order:
        product_id = int(user_item.columns[idx])
        if product_id not in purchased:
            results.append((product_id, float(combined[idx])))
        if len(results) == k:
            break
    return results


def precision_at_k(orders: pd.DataFrame, products: pd.DataFrame,
                   k: int = 5, n_users: int = 200) -> float:
    """Time-based holdout: train on the first 80% of days, test on the rest.

    Similarity is built from all users' training-period orders, so the
    candidate set is the full training catalog -- not just one user's
    own purchases.
    """
    cutoff = orders["timestamp"].quantile(0.8)
    train_orders = orders[orders["timestamp"] < cutoff]
    test_orders = orders[orders["timestamp"] >= cutoff]
    train_item = build_user_item(train_orders)
    cf = item_cf_scores(train_item)
    content_full = content_scores(products)
    col_index = {p: i for i, p in enumerate(products["product_id"])}
    train_cols = [col_index[p] for p in train_item.columns]
    content = content_full[np.ix_(train_cols, train_cols)]
    hits, total, evaluated = 0, 0, 0
    for user_id, group in test_orders.groupby("user_id"):
        if user_id not in train_item.index or evaluated >= n_users:
            continue
        recs = recommend(user_id, train_item, cf, content, k=k)
        recommended = {p for p, _ in recs}
        bought = set(group["product_id"])
        hits += len(recommended & bought)
        total += len(recommended)
        evaluated += 1
    return hits / total if total else 0.0


def main() -> None:
    data = make_data()
    orders = clean_orders(data["orders"])
    user_item = build_user_item(orders)
    cf = item_cf_scores(user_item)
    content = content_scores(data["products"])
    user_id = int(user_item.index[0])
    print(f"Recommendations for user {user_id}:")
    for product_id, score in recommend(user_id, user_item, cf, content, k=5):
        row = data["products"].loc[
            data["products"]["product_id"] == product_id].iloc[0]
        print(f"  product {product_id} ({row['category']}, "
              f"${row['price']:.0f}) score={score:.3f}")
    print(f"\nPrecision@5 (holdout): {precision_at_k(orders, data['products']):.3f}")


if __name__ == "__main__":
    main()

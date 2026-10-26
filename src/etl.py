"""ETL pipeline for the e-commerce platform (demonstration).

Mirrors the production pipeline: transactional systems, clickstream logs,
and customer interactions are ingested, cleaned, and joined into
structured datasets for model training. In production this ran on Python
with Pandas, AWS S3, and AWS Glue; this demo runs the same pipeline shape
on synthetic data in-process.

Run:  python src/etl.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["electronics", "fashion", "home", "beauty", "sports"]
PRODUCT_WORDS = {
    "electronics": "wireless headphones speaker charger laptop phone camera",
    "fashion": "jacket shoes dress shirt sneakers jeans coat",
    "home": "lamp sofa table chair bedsheet cookware curtain",
    "beauty": "serum moisturizer shampoo lipstick cleanser cream",
    "sports": "yoga mat dumbbell shoes bottle racket ball",
}
FEATURE_WORDS = [
    "premium", "compact", "waterproof", "lightweight", "durable", "portable",
    "smart", "eco", "luxury", "budget", "pro", "max", "mini", "ultra",
    "classic", "modern", "soft", "strong", "fast", "quiet", "bright",
    "flexible", "stylish", "heavy-duty", "travel", "outdoor", "wireless",
    "rechargeable", "adjustable", "ergonomic", "sleek", "rugged", "silent",
    "powerful", "gentle", "bold", "natural", "leather", "cotton", "steel",
]


def make_data(n_users: int = 2000, n_products: int = 300,
              n_orders: int = 20000, seed: int = 42) -> dict:
    """Generate synthetic catalog, orders, and clickstream tables."""
    rng = np.random.default_rng(seed)
    products = pd.DataFrame(
        {
            "product_id": np.arange(n_products),
            "category": rng.choice(CATEGORIES, size=n_products),
            "price": np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n_products), 2),
        }
    )
    # Each product gets a unique description: category words plus a few
    # feature words, so content similarity differentiates within a category.
    descriptions = []
    for _, row in products.iterrows():
        feats = " ".join(rng.choice(FEATURE_WORDS, size=3, replace=False))
        descriptions.append(f"{PRODUCT_WORDS[row['category']]} {feats}")
    products["description"] = descriptions

    # Shared taste clusters: 20 groups of users who prefer the same 30
    # products. This creates the co-purchase structure that collaborative
    # filtering learns from, and the shared description words give the
    # content-based side signal.
    by_category = {
        cat: np.where(products["category"].to_numpy() == cat)[0]
        for cat in CATEGORIES
    }
    n_clusters = 20
    cluster_products = []
    cluster_weights = []
    for c in range(n_clusters):
        affinity = CATEGORIES[c % len(CATEGORIES)]
        size = min(30, len(by_category[affinity]))
        preferred = rng.choice(by_category[affinity], size=size, replace=False)
        cluster_products.append(preferred)
        # Zipf-style popularity: a few cluster items are bought far more
        # often, which is what lets the recommender rank them first.
        weights = 1.0 / (1.0 + np.arange(size))
        cluster_weights.append(weights / weights.sum())
        signature = " ".join(rng.choice(FEATURE_WORDS, size=2, replace=False))
        products.loc[preferred, "description"] = (
            products.loc[preferred, "description"] + " " + signature
        )
    user_cluster = rng.integers(0, n_clusters, size=n_users)
    user_ids = rng.integers(0, n_users, size=n_orders)
    from_taste = rng.random(n_orders) < 0.7
    product_ids = rng.integers(0, n_products, size=n_orders)
    for c in range(n_clusters):
        mask = from_taste & (user_cluster[user_ids] == c)
        idx = np.where(mask)[0]
        if len(idx):
            product_ids[idx] = rng.choice(
                cluster_products[c], size=len(idx), p=cluster_weights[c]
            )
    orders = pd.DataFrame(
        {
            "user_id": user_ids,
            "product_id": product_ids,
            "quantity": rng.integers(1, 4, size=n_orders),
            "timestamp": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, size=n_orders), unit="D"),
        }
    )
    orders = orders.merge(
        products[["product_id", "price", "category"]],
        on="product_id", how="left",
    )
    orders["amount"] = orders["price"] * orders["quantity"]

    clicks = pd.DataFrame(
        {
            "user_id": rng.integers(0, n_users, size=n_orders * 2),
            "product_id": rng.integers(0, n_products, size=n_orders * 2),
            "event": rng.choice(["view", "cart", "purchase"],
                                size=n_orders * 2, p=[0.7, 0.2, 0.1]),
        }
    )
    return {"products": products, "orders": orders, "clicks": clicks}


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, duplicates, and invalid rows; parse timestamps."""
    df = orders.dropna().drop_duplicates()
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.reset_index(drop=True)


def build_user_item(orders: pd.DataFrame) -> pd.DataFrame:
    """User-item purchase-count matrix for collaborative filtering."""
    return orders.pivot_table(
        index="user_id", columns="product_id",
        values="quantity", aggfunc="sum", fill_value=0,
    )


def build_user_features(orders: pd.DataFrame,
                        clicks: pd.DataFrame) -> pd.DataFrame:
    """Per-user behavioral features for segmentation."""
    agg = orders.groupby("user_id").agg(
        total_spend=("amount", "sum"),
        orders=("amount", "size"),
        avg_order_value=("amount", "mean"),
        categories=("category", "nunique"),
    )
    clicks_per_user = clicks.groupby("user_id").size().rename("clicks")
    features = agg.join(clicks_per_user, how="left").fillna(0)
    features["clicks"] = features["clicks"].astype(int)
    return features


def main() -> None:
    data = make_data()
    orders = clean_orders(data["orders"])
    user_item = build_user_item(orders)
    features = build_user_features(orders, data["clicks"])
    print(f"Products: {len(data['products']):,}")
    print(f"Orders (clean): {len(orders):,} from {orders['user_id'].nunique():,} users")
    print(f"User-item matrix: {user_item.shape[0]} users x {user_item.shape[1]} products")
    print(f"User features: {features.shape[0]} users, "
          f"mean spend ${features['total_spend'].mean():.0f}")
    print(features.head())


if __name__ == "__main__":
    main()

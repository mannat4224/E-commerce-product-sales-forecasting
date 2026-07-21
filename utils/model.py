"""
CatBoost forecasting model: training, validation, and feature importance,
wrapped for reuse inside the Streamlit app.
"""
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool

FEATURE_COLS = [
    "Month", "Week", "Weekday", "ProductType",
    "KnownPriceMedian", "Quantity_Lag1", "TransactionsPerStockCode",
]
CAT_FEATURES = ["ProductType"]
TARGET_COL = "Quantity"


@st.cache_resource(show_spinner=False)
def train_model(daily_data_hash: str, daily_data: pd.DataFrame, cutoff_week: int,
                 iterations: int = 500, max_depth: int = 6, l2_leaf_reg: int = 3):
    """Trains a CatBoostRegressor on daily_data, splitting train/val by week.
    daily_data_hash is only used as a cache key alongside the params."""
    data = daily_data.copy()

    X = data[FEATURE_COLS].copy()
    y = data[TARGET_COL].copy()
    weeks = data["Week"]

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES]

    x_train, x_val = X.loc[weeks < cutoff_week], X.loc[weeks >= cutoff_week]
    y_train, y_val = y.loc[weeks < cutoff_week], y.loc[weeks >= cutoff_week]

    train_pool = Pool(x_train, y_train, cat_features=cat_idx)
    val_pool = Pool(x_val, y_val, cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=0,
        logging_level="Silent",
        iterations=iterations,
        max_depth=max_depth,
        l2_leaf_reg=l2_leaf_reg,
        od_type="Iter",
        od_wait=40,
        has_time=True,
    )
    model.fit(train_pool, eval_set=val_pool)

    preds = model.predict(x_val)
    results = pd.DataFrame({
        "Actual": y_val.values,
        "Predicted": preds,
    }, index=x_val.index)
    results["AbsError"] = (results.Actual - results.Predicted).abs()
    results["Month"] = data.loc[x_val.index, "Month"]

    metrics = {
        "r2": model.score(val_pool),
        "mae": float(results.AbsError.mean()),
        "median_ae": float(results.AbsError.median()),
        "rmse": float(np.sqrt((results.AbsError ** 2).mean())),
        "tree_count": model.tree_count_,
        "n_train": len(x_train),
        "n_val": len(x_val),
    }

    importances = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.get_feature_importance(train_pool),
    }).sort_values("Importance", ascending=False)

    return model, results, metrics, importances


def predict_single(model, week, weekday, month, product_type, known_price, qty_lag1, transactions):
    row = pd.DataFrame([{
        "Month": month,
        "Week": week,
        "Weekday": weekday,
        "ProductType": str(product_type),
        "KnownPriceMedian": known_price,
        "Quantity_Lag1": qty_lag1,
        "TransactionsPerStockCode": transactions,
    }])[FEATURE_COLS]
    return float(model.predict(row)[0])

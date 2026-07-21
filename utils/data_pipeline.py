"""
Data cleaning + feature engineering pipeline.
Mirrors the exploratory logic from the original notebook, packaged into
reusable, cached functions for the Streamlit app.
"""
import numpy as np
import pandas as pd
import streamlit as st


def _count_numeric_chars(s: str) -> int:
    return sum(1 for c in s if c.isdigit())


def _count_upper_chars(s: str) -> int:
    return sum(1 for c in s if c.isupper())


@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1", dtype={"CustomerID": str})
    return df


@st.cache_data(show_spinner=False)
def clean_data(path: str):
    """Full cleaning pipeline, returns cleaned transaction-level data
    plus a dict of stats describing what was removed at each step."""
    stats = {}
    df = load_raw_data(path)
    stats["raw_rows"] = len(df)
    stats["raw_missing_pct"] = round(df.isnull().sum().sum() / df.size * 100, 2)

    # Drop missing descriptions / customer IDs
    df = df.loc[df.CustomerID.notnull() & df.Description.notnull()].copy()
    stats["after_missing_drop"] = len(df)

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate, cache=True)

    # Remove cancelled invoices (InvoiceNo starting with 'C')
    df["IsCancelled"] = df.InvoiceNo.astype(str).apply(lambda l: l[0] == "C")
    stats["cancelled_pct"] = round(df.IsCancelled.mean() * 100, 2)
    df = df.loc[~df.IsCancelled].drop(columns="IsCancelled")
    stats["after_cancel_drop"] = len(df)

    # Keep only "clean" 5-digit numeric stock codes
    df["StockCode"] = df.StockCode.astype(str)
    stock_len = df.StockCode.apply(len)
    stock_numeric = df.StockCode.apply(_count_numeric_chars)
    df = df.loc[(stock_numeric == 5) & (stock_len == 5)].copy()
    stats["after_stockcode_filter"] = len(df)

    # Filter out low-signal (mostly-lowercase / admin) descriptions
    up_chars = df.Description.apply(_count_upper_chars)
    df = df.loc[up_chars > 5].copy()
    stats["after_description_filter"] = len(df)

    # Price / quantity sanity filters (removes zero/negative price, extreme outliers)
    df = df.loc[df.UnitPrice > 0].copy()
    price_hi = np.quantile(df.UnitPrice, 0.95)
    df = df.loc[(df.UnitPrice > 0.1) & (df.UnitPrice < 20)].copy()
    stats["after_price_filter"] = len(df)

    qty_hi = np.quantile(df.Quantity, 0.95)
    df = df.loc[df.Quantity < 55].copy()
    stats["after_quantity_filter"] = len(df)

    df["Revenue"] = df.Quantity * df.UnitPrice
    df["Year"] = df.InvoiceDate.dt.year
    df["Quarter"] = df.InvoiceDate.dt.quarter
    df["Month"] = df.InvoiceDate.dt.month
    df["Week"] = df.InvoiceDate.dt.isocalendar().week.astype(int)
    df["Weekday"] = df.InvoiceDate.dt.weekday
    df["Day"] = df.InvoiceDate.dt.day
    df["Dayofyear"] = df.InvoiceDate.dt.dayofyear
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["DescriptionLength"] = df.Description.astype(str).str.len()

    stats["final_rows"] = len(df)
    stats["date_min"] = df.InvoiceDate.min()
    stats["date_max"] = df.InvoiceDate.max()
    stats["n_products"] = df.StockCode.nunique()
    stats["n_customers"] = df.CustomerID.nunique()
    stats["n_countries"] = df.Country.nunique()

    return df, stats


@st.cache_data(show_spinner=False)
def build_daily_product_data(path: str, n_clusters: int = 10, cutoff_week: int = 45):
    """Builds the daily-per-product modeling table: quantity/revenue
    aggregated by day+stockcode, enriched with product-cluster and
    price/attraction features, matching the notebook's feature set."""
    df, _ = clean_data(path)

    # --- Product-level features (computed only on data before cutoff, to avoid leakage) ---
    hist = df.loc[df.Week < cutoff_week]
    products = pd.DataFrame(index=hist.StockCode.unique())
    products["MedianPrice"] = hist.groupby("StockCode").UnitPrice.median()
    products["MedianQuantities"] = hist.groupby("StockCode").Quantity.median()
    products["Customers"] = hist.groupby("StockCode").CustomerID.nunique()
    products["DescriptionLength"] = hist.groupby("StockCode").DescriptionLength.median()
    products = products.dropna()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X_prod = StandardScaler().fit_transform(products.values)
    k = min(n_clusters, max(2, len(products) // 20))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    products["cluster"] = km.fit_predict(X_prod)

    # --- Daily aggregation per product ---
    daily = df.groupby(["Date", "StockCode"]).agg(
        Quantity=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
    ).reset_index()
    daily["Year"] = daily.Date.dt.year
    daily["Month"] = daily.Date.dt.month
    daily["Week"] = daily.Date.dt.isocalendar().week.astype(int)
    daily["Weekday"] = daily.Date.dt.weekday
    daily["ProductType"] = daily.StockCode.map(products["cluster"]).fillna(-1).astype(int).astype(str)

    # Known price stats per product (weekly)
    price_stats = df.groupby(["StockCode", "Week"]).UnitPrice.median().rename("KnownPriceMedian")
    daily = daily.merge(price_stats, on=["StockCode", "Week"], how="left")

    # Weekly "attraction" (unique customers) and mean price per weekday/product-type, lagged
    attraction = df.groupby(["Week", "Weekday", "ProductType"] if "ProductType" in df.columns
                             else ["Week", "Weekday"]).size()

    # Simple, leakage-safe lag: previous week's median quantity for same stockcode
    daily = daily.sort_values(["StockCode", "Date"])
    daily["Quantity_Lag1"] = daily.groupby("StockCode")["Quantity"].shift(1)
    daily["TransactionsPerStockCode"] = daily.StockCode.map(
        hist.groupby("StockCode").InvoiceNo.nunique()
    )

    daily = daily.dropna(subset=["Quantity_Lag1", "KnownPriceMedian"])

    return daily, products, stats_summary(df)


def stats_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "products": df.StockCode.nunique(),
        "customers": df.CustomerID.nunique(),
        "countries": df.Country.nunique(),
        "date_min": df.InvoiceDate.min(),
        "date_max": df.InvoiceDate.max(),
    }

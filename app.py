import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_pipeline import clean_data, build_daily_product_data
from utils.model import train_model, predict_single, FEATURE_COLS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "data.csv")
CUTOFF_WEEK = 45  # last ~7 weeks of the year held out for validation

st.set_page_config(
    page_title="Retail Sales Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar navigation ----------
st.sidebar.title("📈 Sales Forecast Studio")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Cleaning", "Exploratory Analysis", "Product Clusters",
     "Forecasting Model", "Try a Prediction"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "An end-to-end pipeline: raw UK online-retail transactions → cleaned "
    "daily product sales → CatBoost quantity forecast."
)


@st.cache_data(show_spinner=False)
def get_clean():
    return clean_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_daily(n_clusters, cutoff_week):
    return build_daily_product_data(DATA_PATH, n_clusters=n_clusters, cutoff_week=cutoff_week)


# ============================================================
if page == "Overview":
    st.title("E-Commerce Sales Forecasting")
    st.markdown(
        "This project forecasts **daily product-level sales quantity** for an "
        "online retailer, using the classic UK Online Retail transaction dataset."
    )

    df, stats = get_clean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean transactions", f"{stats['final_rows']:,}")
    c2.metric("Unique products", f"{stats['n_products']:,}")
    c3.metric("Unique customers", f"{stats['n_customers']:,}")
    c4.metric("Countries", f"{stats['n_countries']:,}")

    st.markdown("### Pipeline")
    st.markdown(
        """
1. **Data Cleaning** — drop missing IDs/descriptions, remove cancellations, keep valid 5-digit product codes, filter price/quantity outliers.
2. **Exploratory Analysis** — understand product, customer, and country patterns.
3. **Product Clusters** — group products by price/demand/customer-reach profile (KMeans) so the model can generalize across thousands of SKUs.
4. **Forecasting Model** — CatBoost regressor predicts daily quantity sold per product, validated on the most recent weeks (time-based split, no leakage).
5. **Try a Prediction** — interactively query the trained model.
        """
    )
    st.info(
        f"Data spans **{stats['date_min'].date()}** to **{stats['date_max'].date()}**. "
        f"Model validates on weeks ≥ {CUTOFF_WEEK}, training only on earlier weeks."
    )

# ============================================================
elif page == "Data Cleaning":
    st.title("Data Cleaning Pipeline")
    df, stats = get_clean()

    funnel_labels = [
        "Raw rows", "After dropping missing IDs/descriptions", "After removing cancellations",
        "After stock-code filter", "After description filter", "After price filter",
        "After quantity filter",
    ]
    funnel_values = [
        stats["raw_rows"], stats["after_missing_drop"], stats["after_cancel_drop"],
        stats["after_stockcode_filter"], stats["after_description_filter"],
        stats["after_price_filter"], stats["after_quantity_filter"],
    ]
    fig = go.Figure(go.Funnel(y=funnel_labels, x=funnel_values, textinfo="value+percent initial"))
    fig.update_layout(height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Raw missing-value cells", f"{stats['raw_missing_pct']}%")
    col2.metric("Cancelled orders removed", f"{stats['cancelled_pct']}%")

    st.markdown("### Cleaned sample")
    st.dataframe(df.sample(min(200, len(df)), random_state=0).sort_values("InvoiceDate"),
                 use_container_width=True, height=300)

# ============================================================
elif page == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    df, stats = get_clean()

    tab1, tab2, tab3, tab4 = st.tabs(["Top Products", "Countries", "Price & Quantity", "Time Trend"])

    with tab1:
        top_desc = df.Description.value_counts().head(15).sort_values()
        fig = px.bar(x=top_desc.values, y=top_desc.index, orientation="h",
                     labels={"x": "Order lines", "y": ""}, title="Most frequently ordered products")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_country = df.Country.value_counts().head(15)
        fig = px.bar(x=top_country.index, y=top_country.values, log_y=True,
                     labels={"x": "", "y": "Order lines (log scale)"}, title="Orders by country")
        st.plotly_chart(fig, use_container_width=True)
        uk_pct = (df.Country == "United Kingdom").mean() * 100
        st.caption(f"United Kingdom accounts for {uk_pct:.1f}% of all order lines.")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="UnitPrice", nbins=40, title="Unit price distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="Quantity", nbins=40, title="Quantity distribution", log_y=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        daily_rev = df.groupby("Date").Revenue.sum().reset_index()
        fig = px.line(daily_rev, x="Date", y="Revenue", title="Daily revenue")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
elif page == "Product Clusters":
    st.title("Product Clustering")
    st.caption("Products are grouped by price, typical order quantity, customer reach, "
               "and description length, so the forecasting model can share patterns "
               "across similar products instead of memorizing thousands of individual SKUs.")

    n_clusters = st.slider("Number of clusters", 4, 20, 10)
    daily, products, dstats = get_daily(n_clusters, CUTOFF_WEEK)

    fig = px.scatter(
        products, x="MedianPrice", y="MedianQuantities", color=products["cluster"].astype(str),
        size="Customers", hover_data=["DescriptionLength"],
        labels={"color": "Cluster"}, title="Product clusters (price vs typical order quantity)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster profile")
    profile = products.groupby("cluster")[["MedianPrice", "MedianQuantities", "Customers", "DescriptionLength"]].mean().round(2)
    profile["n_products"] = products.groupby("cluster").size()
    st.dataframe(profile, use_container_width=True)

# ============================================================
elif page == "Forecasting Model":
    st.title("Forecasting Model")
    st.caption("Gradient-boosted trees (CatBoost) predicting daily quantity sold per product. "
               "Validated on a held-out block of the most recent weeks — the model never sees "
               "future weeks during training.")

    with st.expander("Model settings", expanded=False):
        n_clusters = st.slider("Product clusters (feature input)", 4, 20, 10, key="mdl_clusters")
        iterations = st.slider("Boosting iterations", 100, 1000, 500, step=100)
        max_depth = st.slider("Tree max depth", 3, 10, 6)

    daily, products, dstats = get_daily(n_clusters, CUTOFF_WEEK)
    data_hash = f"{len(daily)}-{n_clusters}-{iterations}-{max_depth}"

    with st.spinner("Training CatBoost model..."):
        model, results, metrics, importances = train_model(
            data_hash, daily, CUTOFF_WEEK, iterations=iterations, max_depth=max_depth,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (validation)", f"{metrics['r2']:.3f}")
    c2.metric("Median absolute error", f"{metrics['median_ae']:.2f} units")
    c3.metric("RMSE", f"{metrics['rmse']:.2f} units")
    c4.metric("Trees used", metrics["tree_count"])
    st.caption(f"Trained on {metrics['n_train']:,} rows, validated on {metrics['n_val']:,} held-out rows.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(results, x="Predicted", y="Actual", color="AbsError",
                          color_continuous_scale="RdYlBu_r", opacity=0.6,
                          title="Predicted vs actual (validation set)")
        max_v = max(results.Actual.max(), results.Predicted.max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                      line=dict(color="black", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                     title="Feature importance")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    monthly_rmse = results.groupby("Month").apply(lambda g: np.sqrt((g.AbsError ** 2).mean()))
    fig = px.line(x=monthly_rmse.index, y=monthly_rmse.values, markers=True,
                  labels={"x": "Month", "y": "RMSE"}, title="Validation RMSE by month")
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["trained_model"] = model
    st.session_state["daily_data"] = daily
    st.session_state["products"] = products

# ============================================================
elif page == "Try a Prediction":
    st.title("Try a Prediction")
    st.caption("Pick a product and a date context — the trained model estimates how many units will sell.")

    n_clusters = 10
    daily, products, dstats = get_daily(n_clusters, CUTOFF_WEEK)
    data_hash = f"{len(daily)}-{n_clusters}-500-6"

    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
    else:
        with st.spinner("Training CatBoost model..."):
            model, results, metrics, importances = train_model(data_hash, daily, CUTOFF_WEEK)

    stock_codes = sorted(daily.StockCode.unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_code = st.selectbox("Stock code", stock_codes, index=0)
        product_rows = daily.loc[daily.StockCode == stock_code]
        product_type = product_rows.ProductType.iloc[-1] if len(product_rows) else "0"
    with col2:
        month = st.slider("Month", 1, 12, 11)
        weekday = st.selectbox("Weekday", list(range(7)),
                                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d])
    with col3:
        week = st.slider("Week of year", 1, 52, 46)

    default_price = float(product_rows.KnownPriceMedian.median()) if len(product_rows) else 3.0
    default_lag = float(product_rows.Quantity_Lag1.median()) if len(product_rows) else 5.0
    default_txn = float(product_rows.TransactionsPerStockCode.median()) if len(product_rows) else 20.0

    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Known unit price", value=round(default_price, 2), step=0.1)
    lag_qty = c2.number_input("Previous day's quantity (lag)", value=round(default_lag, 1), step=1.0)
    txns = c3.number_input("Historical transactions for this product", value=float(round(default_txn)), step=1.0)
    if st.button("Predict quantity", type="primary"):
        pred = predict_single(model, week, weekday, month, product_type, price, lag_qty, txns)
        st.success(f"Predicted units sold: **{max(0, pred):.1f}**")


# Retail Sales Forecast Studio

An end-to-end e-commerce sales forecasting project, built as an interactive
Streamlit dashboard. It takes the raw UK Online Retail transaction log,
cleans it, explores it, clusters products, and trains a CatBoost model to
forecast daily product-level sales quantity — with a live prediction tool.

This turns the original analysis notebook into a presentable, clickable
application instead of a static script.

## Pages

- **Overview** — project summary and key dataset stats
- **Data Cleaning** — funnel chart of every filtering step, with row counts
- **Exploratory Analysis** — top products, countries, price/quantity
  distributions, revenue over time
- **Product Clusters** — KMeans clustering of products by price/demand/reach
- **Forecasting Model** — trains CatBoost with a time-based train/validation
  split, shows R², RMSE, feature importance, predicted-vs-actual plot
- **Try a Prediction** — pick a product and date context, get a live
  predicted quantity

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

The first load will take a little longer while the data is cleaned and the
model trains — after that, Streamlit's caching keeps it fast.

## Project structure

```
sales_dashboard/
├── app.py                  # Main Streamlit app (pages + UI)
├── utils/
│   ├── data_pipeline.py    # Cleaning + feature engineering
│   └── model.py            # CatBoost training/prediction
├── data/
│   └── data.csv            # UK Online Retail dataset
├── requirements.txt
└── .streamlit/config.toml  # Theme
```

## Deploying it for free (Streamlit Community Cloud)

This gets you a public link you can put in your evaluation report.

1. Create a free GitHub account if you don't have one, and create a new
   **public** repository (e.g. `sales-forecast-dashboard`).
2. Upload this entire folder's contents to that repository (drag-and-drop
   on github.com works fine, or `git push` if you're comfortable with git).
3. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with your GitHub account.
4. Click **"New app"**, pick your repository, branch `main`, and set the
   main file path to `app.py`.
5. Click **Deploy**. The first build takes a few minutes (installing
   catboost etc.). You'll get a link like
   `https://your-app-name.streamlit.app` — that's your deployment link.

**Note on the data file:** `data/data.csv` is ~45 MB, which is within
GitHub's normal file size limits, so a plain upload/push works without
Git LFS.

## Notes for your evaluation

- The model is validated on a **held-out block of the most recent weeks**
  (not a random split), so the reported R²/RMSE reflect genuine
  forward-looking forecast accuracy rather than leakage from shuffling.
- Product clustering exists because there are thousands of individual
  SKUs — clustering lets the model learn shared demand patterns across
  similar products rather than treating each one as unrelated.
- Feature importance and the predicted-vs-actual plot are there so you can
  explain *why* the model predicts what it does, not just that it does.

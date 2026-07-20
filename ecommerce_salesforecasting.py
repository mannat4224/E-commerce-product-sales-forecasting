!pip install catboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from catboost import CatBoostRegressor, Pool, cv
from catboost import MetricVisualizer

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.stats import boxcox
from os import listdir

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import shap
shap.initjs()
print(listdir('/content/sample_data'))
data = pd.read_csv('/content/sample_data/california_housing_train.csv', encoding='ISO-8859-1')
data.shape
data = pd.read_csv("/content/data.csv", encoding="ISO-8859-1", dtype={'CustomerID': str})
data.shape

2. Get familiar with the data
data.head()
3. Get an initial feeling for the data by exploration
Missing values
missing_percentage = data.isnull().sum() / data.shape[0] * 100
missing_percentage
data[data.Description.isnull()].head()
data[data.Description.isnull()].CustomerID.isnull().value_counts()
data[data.Description.isnull()].UnitPrice.value_counts()
Missing Customer IDs
data[data.CustomerID.isnull()].head()
data.loc[data.CustomerID.isnull(), ["UnitPrice", "Quantity"]].describe()
Hidden missing descriptions
data.loc[data.Description.isnull()==False, "lowercase_descriptions"] = data.loc[
    data.Description.isnull()==False,"Description"
].apply(lambda l: l.lower())

data.lowercase_descriptions.dropna().apply(
    lambda l: np.where("nan" in l, True, False)
).value_counts()

data.lowercase_descriptions.dropna().apply(
    lambda l: np.where("" == l, True, False)
).value_counts()
data.loc[data.lowercase_descriptions.isnull()==False, "lowercase_descriptions"] = data.loc[
    data.lowercase_descriptions.isnull()==False, "lowercase_descriptions"
].apply(lambda l: np.where("nan" in l, None, l))
data = data.loc[(data.CustomerID.isnull()==False) & (data.lowercase_descriptions.isnull()==False)].copy()
data.isnull().sum().sum()
data["InvoiceDate"] = pd.to_datetime(data.InvoiceDate, cache=True)

data.InvoiceDate.max() - data.InvoiceDate.min()
print("Datafile starts with timepoint {}".format(data.InvoiceDate.min()))
print("Datafile ends with timepoint {}".format(data.InvoiceDate.max()))
The invoice number
data.InvoiceNo.nunique()
data["IsCancelled"]=np.where(data.InvoiceNo.apply(lambda l: l[0]=="C"), True, False)
data.IsCancelled.value_counts() / data.shape[0] * 100
data.loc[data.IsCancelled==True].describe()
data = data.loc[data.IsCancelled==False].copy()
data = data.drop("IsCancelled", axis=1)
Stockcodes
data.StockCode.nunique()
stockcode_counts = data.StockCode.value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(2,1,figsize=(20,15))
sns.barplot(x=stockcode_counts.iloc[0:20].index,
            y=stockcode_counts.iloc[0:20].values,
            ax = ax[0], palette="Oranges_r")
ax[0].set_ylabel("Counts")
ax[0].set_xlabel("Stockcode")
ax[0].set_title("Which stockcodes are most common?");
sns.distplot(np.round(stockcode_counts/data.shape[0]*100,2),
             kde=False,
             bins=20,
             ax=ax[1], color="Orange")
ax[1].set_title("How seldom are stockcodes?")
ax[1].set_xlabel("% of data with this stockcode")
ax[1].set_ylabel("Frequency");
Let's count the number of numeric chars in and the length of the stockcode:

def count_numeric_chars(l):
    return sum(1 for c in l if c.isdigit())

data["StockCodeLength"] = data.StockCode.apply(lambda l: len(l))
data["nNumericStockCode"] = data.StockCode.apply(lambda l: count_numeric_chars(l))
Even though the majority of samples has a stockcode that consists of 5 numeric chars, we can see that there are other occurences as well. The length can vary between 1 and 12 and there are stockcodes with no numeric chars at all!
data.loc[data.nNumericStockCode < 5].lowercase_descriptions.value_counts()
data.loc[data.nNumericStockCode < 5].lowercase_descriptions.value_counts()
data = data.loc[(data.nNumericStockCode == 5) & (data.StockCodeLength==5)].copy()
data.StockCode.nunique()
data = data.drop(["nNumericStockCode", "StockCodeLength"], axis=1)
Descriptions
data.Description.nunique()
description_counts = data.Description.value_counts().sort_values(ascending=False).iloc[0:30]
plt.figure(figsize=(20,5))
sns.barplot(x=description_counts.index, y=description_counts.values, palette="Purples_r")
plt.ylabel("Counts")
plt.title("Which product descriptions are most common?");
plt.xticks(rotation=90);
def count_lower_chars(l):
    return sum(1 for c in l if c.islower())
def count_upper_chars(l):
    return sum(1 for c in l if c.isupper())

data["UpCharsInDescription"] = data.Description.apply(lambda l: count_upper_chars(l))
data["LowCharsInDescription"] = data.Description.apply(lambda l: count_lower_chars(l))
data.UpCharsInDescription.describe()
data.loc[data.UpCharsInDescription <=5].Description.value_counts()
data = data.loc[data.UpCharsInDescription > 5].copy()
data["DescriptionLength"] = data.Description.apply(lambda l: len(l))
dlength_counts = data.loc[data.DescriptionLength < 14].Description.value_counts()

plt.figure(figsize=(20,5))
sns.barplot(x=dlength_counts.index, y=dlength_counts.values, palette="Purples_r")
plt.xticks(rotation=90);
data.StockCode.nunique()
data.Description.nunique()
data.groupby("StockCode").Description.nunique().sort_values(ascending=False).iloc[0:10]
data.loc[data.StockCode == "23244"].Description.value_counts()
Customers
data.CustomerID.nunique()
customer_counts = data.CustomerID.value_counts().sort_values(ascending=False).iloc[0:20]
plt.figure(figsize=(20,5))
sns.barplot(x=customer_counts.index, y=customer_counts.values, order=customer_counts.index)
plt.ylabel("Counts")
plt.xlabel("CustomerID")
plt.title("Which customers are most common?");
#plt.xticks(rotation=90);
Countries
data.Country.nunique()
country_counts = data.Country.value_counts().sort_values(ascending=False).iloc[0:20]
plt.figure(figsize=(20,5))
sns.barplot(x=country_counts.index, y=country_counts.values, palette="Greens_r")
plt.ylabel("Counts")
plt.title("Which countries made the most transactions?");
plt.xticks(rotation=90);
plt.yscale("log")
data.loc[data.Country=="United Kingdom"].shape[0] / data.shape[0] * 100
data["UK"] = np.where(data.Country == "United Kingdom", 1, 0)
Unit Price
data.UnitPrice.describe()
data.loc[data.UnitPrice == 0].sort_values(by="Quantity", ascending=False).head()
data = data.loc[data.UnitPrice > 0].copy()
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(data.UnitPrice, ax=ax[0], kde=False, color="red")
sns.distplot(np.log(data.UnitPrice), ax=ax[1], bins=20, color="tomato", kde=False)
ax[1].set_xlabel("Log-Unit-Price");
np.exp(-2)
np.exp(3)
np.quantile(data.UnitPrice, 0.95)
data = data.loc[(data.UnitPrice > 0.1) & (data.UnitPrice < 20)].copy()
Quantities
data.Quantity.describe()
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(data.Quantity, ax=ax[0], kde=False, color="limegreen");
sns.distplot(np.log(data.Quantity), ax=ax[1], bins=20, kde=False, color="limegreen");
ax[0].set_title("Quantity distribution")
ax[0].set_yscale("log")
ax[1].set_title("Log-Quantity distribution")
ax[1].set_xlabel("Natural-Log Quantity");
np.exp(4)
np.quantile(data.Quantity, 0.95)
data = data.loc[data.Quantity < 55].copy()
Conclusion
Focus on daily product sales
As we like to predict the daily amount of product sales, we need to compute a daily aggregation of this data. For this purpose we need to extract temporal features out of the InvoiceDate. In addition we can compute the revenue gained by a transaction using the unit price and the quantity:
data["Revenue"] = data.Quantity * data.UnitPrice

data["Year"] = data.InvoiceDate.dt.year
data["Quarter"] = data.InvoiceDate.dt.quarter
data["Month"] = data.InvoiceDate.dt.month
data["Week"] = data.InvoiceDate.dt.isocalendar().week
data["Weekday"] = data.InvoiceDate.dt.weekday
data["Day"] = data.InvoiceDate.dt.day
data["Dayofyear"] = data.InvoiceDate.dt.dayofyear
data["Date"] = pd.to_datetime(data[['Year', 'Month', 'Day']])
As the key task of this kernel is to predict the amount of products sold per day, we can sum up the daily quantities per product stockcode :
grouped_features = ["Date", "Year", "Quarter","Month", "Week", "Weekday", "Dayofyear", "Day",
                    "StockCode"]
This way we loose information abount customers, countries and price information but we will recover it later on during this kernel. Besides the quantities let's aggregate the revenues as well:
daily_data = pd.DataFrame(data.groupby(grouped_features).Quantity.sum(),
                          columns=["Quantity"])
daily_data["Revenue"] = data.groupby(grouped_features).Revenue.sum()
daily_data = daily_data.reset_index()
daily_data.head(5)
How are the quantities and revenues distributed?
daily_data.loc[:, ["Quantity", "Revenue"]].describe()
low_quantity = daily_data.Quantity.quantile(0.01)
high_quantity = daily_data.Quantity.quantile(0.99)
print((low_quantity, high_quantity))
low_revenue = daily_data.Revenue.quantile(0.01)
high_revenue = daily_data.Revenue.quantile(0.99)
print((low_revenue, high_revenue))
Let's only use target ranges data that are occupied by 90 % of the data entries. This is a first and easy strategy to exclude heavy outliers but we should always be aware of the fact that we have lost some information given by the remaining % we have excluded. It could be nice and useful in general to understand and analyse what has caused these outliers.
samples = daily_data.shape[0]
daily_data = daily_data.loc[
    (daily_data.Quantity >= low_quantity) & (daily_data.Quantity <= high_quantity)]
daily_data = daily_data.loc[
    (daily_data.Revenue >= low_revenue) & (daily_data.Revenue <= high_revenue)]

How much entries have we lost?
samples - daily_data.shape[0]
Let's take a look at the remaining distributions of daily quantities:
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(daily_data.Quantity.values, kde=True, ax=ax[0], color="Orange", bins=30);
sns.distplot(np.log(daily_data.Quantity.values), kde=True, ax=ax[1], color="Orange", bins=30);
ax[0].set_xlabel("Number of daily product sales");
ax[0].set_ylabel("Frequency");
ax[0].set_title("How many products are sold per day?");

Catboost family and hyperparameter class
To easily generate new models and compare results between them I wrote some classes:
Hyperparameter Class

This class holds all important hyperparameters we have to set before training like the loss function, the evaluation metric, the max depth of trees, the number of max number of trees (iterations) and the l2_leaf_reg for regularization to avoid overfitting.
class CatHyperparameter:

    def __init__(self,
                 loss="RMSE",
                 metric="RMSE",
                 iterations=1000,
                 max_depth=4,
                 l2_leaf_reg=3,
                 #learning_rate=0.5,
                 seed=0):
        self.loss = loss,
        self.metric = metric,
        self.max_depth = max_depth,
        self.l2_leaf_reg = l2_leaf_reg,
        #self.learning_rate = learning_rate,
        self.iterations=iterations
        self.seed = seed
Catmodel class

This model obtains a train & validation pool as data or pandas dataframes for features X and targets y together with a week. It's the first week of our validation data and all other weeks above are used as well. It trains the model and can show learning process as well as feature importances and some figures for result analysis. It's the fastest choice you can make for playing around.
class Catmodel:

    def __init__(self, name, params):
        self.name = name
        self.params = params

    def set_data_pool(self, train_pool, val_pool):
        self.train_pool = train_pool
        self.val_pool = val_pool

    def set_data(self, X, y, week):
        cat_features_idx = np.where(X.dtypes != np.float)[0]
        x_train, self.x_val = X.loc[X.Week < week], X.loc[X.Week >= week]
        y_train, self.y_val = y.loc[X.Week < week], y.loc[X.Week >= week]
        self.train_pool = Pool(x_train, y_train, cat_features=cat_features_idx)
        self.val_pool = Pool(self.x_val, self.y_val, cat_features=cat_features_idx)

    def prepare_model(self):
        self.model = CatBoostRegressor(
                loss_function = self.params.loss[0],
                random_seed = self.params.seed,
                logging_level = 'Silent',
                iterations = self.params.iterations,
                max_depth = self.params.max_depth[0],
                #learning_rate = self.params.learning_rate[0],
                l2_leaf_reg = self.params.l2_leaf_reg[0],
                od_type='Iter',
                od_wait=40,
                train_dir=self.name,
                has_time=True
            )

    def learn(self, plot=False):
        self.prepare_model()
        self.model.fit(self.train_pool, eval_set=self.val_pool, plot=plot);
        print("{}, early-stopped model tree count {}".format(
            self.name, self.model.tree_count_
        ))

    def score(self):
        return self.model.score(self.val_pool)

    def show_importances(self, kind="bar"):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.val_pool)
        if kind=="bar":
            return shap.summary_plot(shap_values, self.x_val, plot_type="bar")
        return shap.summary_plot(shap_values, self.x_val)

    def get_val_results(self):
        self.results = pd.DataFrame(self.y_val)
        self.results["prediction"] = self.predict(self.x_val)
        self.results["error"] = np.abs(
            self.results[self.results.columns.values[0]].values - self.results.prediction)
        self.results["Month"] = self.x_val.Month
        self.results["SquaredError"] = self.results.error.apply(lambda l: np.power(l, 2))

    def show_val_results(self):
        self.get_val_results()
        fig, ax = plt.subplots(1,2,figsize=(20,5))
        sns.distplot(self.results.error, ax=ax[0])
        ax[0].set_xlabel("Single absolute error")
        ax[0].set_ylabel("Density")
        self.median_absolute_error = np.median(self.results.error)
        print("Median absolute error: {}".format(self.median_absolute_error))
        ax[0].axvline(self.median_absolute_error, c="black")
        ax[1].scatter(self.results.prediction.values,
                      self.results[self.results.columns[0]].values,
                      c=self.results.error, cmap="RdYlBu_r", s=1)
        ax[1].set_xlabel("Prediction")
        ax[1].set_ylabel("Target")
        return ax

    def get_monthly_RMSE(self):
        return self.results.groupby("Month").SquaredError.mean().apply(lambda l: np.sqrt(l))

    def predict(self, x):
        return self.model.predict(x)

    def get_dependence_plot(self, feature1, feature2=None):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.val_pool)
        if feature2 is None:
            return shap.dependence_plot(
                feature1,
                shap_values,
                self.x_val,
            )
        else:
            return shap.dependence_plot(
                feature1,
                shap_values,
                self.x_val,
                interaction_index=feature2
            )

Feature engineering
# RESTORE DATA STATE
import pandas as pd
import numpy as np

print("Restoring data state...")
data = pd.read_csv("/content/data.csv", encoding="ISO-8859-1", dtype={'CustomerID': str})
data["InvoiceDate"] = pd.to_datetime(data.InvoiceDate, cache=True)
data["Week"] = data.InvoiceDate.dt.isocalendar().week
data['DescriptionLength'] = data['Description'].astype(str).str.len()
# Ensure week variable exists
week = 50
print("Data loaded and basic features restored. You can now run the aggregation cell.")
import pandas as pd
import numpy as np

# Aggregating product-level features for weeks prior to the cutoff
products = pd.DataFrame(index=data.loc[data.Week < week].StockCode.unique(), columns=["MedianPrice"])

products["MedianPrice"] = data.loc[data.Week < week].groupby("StockCode").UnitPrice.median()
products["MedianQuantities"] = data.loc[data.Week < week].groupby("StockCode").Quantity.median()
products["Customers"] = data.loc[data.Week < week].groupby("StockCode").CustomerID.nunique()
products["DescriptionLength"] = data.loc[data.Week < week].groupby("StockCode").DescriptionLength.median()

org_cols = np.copy(products.columns.values)
print("Success: Product aggregation complete.")
products.head()
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import numpy as np

# Filter for positive values to satisfy Box-Cox requirements
plot_df = products[(products.MedianPrice > 0) & (products.MedianQuantities > 0)].copy()

# Applying Box-Cox transformation
price_transformed, _ = boxcox(plot_df.MedianPrice.values)
quantity_transformed, _ = boxcox(plot_df.MedianQuantities.values)

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Scatter plot using transformed values
scatter = ax[0].scatter(price_transformed, quantity_transformed,
                        c=plot_df.Customers.values, cmap="coolwarm_r")

ax[0].set_xlabel("Boxcox-Median-UnitPrice")
ax[0].set_ylabel("Boxcox-Median-Quantities")
ax[0].set_title("Price vs Quantity (Transformed)")

# Add a colorbar to show the number of customers
plt.colorbar(scatter, ax=ax[0], label='Customers')

plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler

X = products.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.cluster import KMeans

# Perform clustering on product features
km = KMeans(n_clusters=30, random_state=42)
products["cluster"] = km.fit_predict(X)

# Calculate Revenue in the base 'data' DataFrame
data["Revenue"] = data["Quantity"] * data["UnitPrice"]

# Re-create daily_data aggregation
grouped_features = ["InvoiceDate", "StockCode"]
daily_data = data.groupby(grouped_features).agg({
    "Quantity": "sum",
    "Revenue": "sum"
}).reset_index()

# Map the clusters (ProductType) back to the daily transaction data
daily_data["ProductType"] = daily_data.StockCode.map(products.cluster)
daily_data.ProductType = daily_data.ProductType.astype("object")

# Display results
print(f"Daily data shape: {daily_data.shape}")
daily_data.head()
Baseline for product types
daily_data["KnownStockCodeUnitPriceMedian"] = daily_data.StockCode.map(
    data.groupby("StockCode").UnitPrice.median())

known_price_iqr = data.groupby("StockCode").UnitPrice.quantile(0.75)
known_price_iqr -= data.groupby("StockCode").UnitPrice.quantile(0.25)
daily_data["KnownStockCodeUnitPriceIQR"] = daily_data.StockCode.map(known_price_iqr)
# The previous 'index' rename put row numbers into StockCode instead of actual codes
# Let's re-generate daily_data correctly from the original data

# Ensure data has proper types and temporal features
data['StockCode'] = data['StockCode'].astype(str)
data['Year'] = data.InvoiceDate.dt.year
data['Month'] = data.InvoiceDate.dt.month
data['Week'] = data.InvoiceDate.dt.isocalendar().week.astype(int)
data['Weekday'] = data.InvoiceDate.dt.weekday

# 1. Re-aggregate daily_data properly to ensure StockCode is a column
grouped_features = ["InvoiceDate", "StockCode"]
daily_data = data.groupby(grouped_features).agg({
    "Quantity": "sum",
    "Revenue": "sum"
}).reset_index()

# 2. Re-apply temporal features to daily_data
daily_data['Year'] = daily_data.InvoiceDate.dt.year
daily_data['Month'] = daily_data.InvoiceDate.dt.month
daily_data['Week'] = daily_data.InvoiceDate.dt.isocalendar().week.astype(int)
daily_data['Weekday'] = daily_data.InvoiceDate.dt.weekday

# 3. Calculate statistics from raw 'data'
to_group = ["StockCode", "Year", "Month", "Week", "Weekday"]
stats = data.groupby(to_group).UnitPrice.agg([
    ('KnownStockCodePrice_WW_median', 'median'),
    ('KnownStockCodePrice_WW_mean', 'mean'),
    ('KnownStockCodePrice_WW_std', 'std')
]).reset_index()

# 4. Final Merge
daily_data = daily_data.merge(stats, on=to_group, how='left')

# 5. Restore Cluster (ProductType) mapping
if 'products' in globals():
    daily_data["ProductType"] = daily_data.StockCode.map(products.cluster)

print(f"Merge successful. Sample of mapped prices:\n{daily_data[to_group + ['KnownStockCodePrice_WW_median']].head()}")
daily_data.head()
daily_data.head()
About countries and customers
delta = 1
to_group = ["Week", "Weekday", "ProductType"]

# Ensure ProductType is mapped to the transactional data for the groupby
if 'ProductType' not in data.columns and 'products' in globals():
    data["ProductType"] = data.StockCode.map(products.cluster)

# Reset index in case the cell was partially run before, ensuring columns are accessible
daily_data = daily_data.reset_index()
# Drop any duplicate index columns that might have been created (like 'level_0')
daily_data = daily_data.loc[:, ~daily_data.columns.str.contains('^level_')]

# Perform the aggregation from 'data' to get attraction and price metrics
# We use the index of daily_data to align the results
temp_stats = data.groupby(to_group).agg(
    WeekWeekdayAttraction=('CustomerID', 'nunique'),
    WeekWeekdayMeanUnitPrice=('UnitPrice', lambda x: np.round(x.mean(), 2))
)

# Set index on daily_data to match the stats and join
daily_data = daily_data.set_index(to_group)
daily_data = daily_data.join(temp_stats)

# Create the Lagged features
daily_data["WeekWeekdayAttraction_Lag1"] = daily_data["WeekWeekdayAttraction"].shift(1)
daily_data["WeekWeekdayMeanUnitPrice_Lag1"] = daily_data["WeekWeekdayMeanUnitPrice"].shift(1)

# Reset index and clean up based on the provided logic
daily_data = daily_data.reset_index()
daily_data.loc[daily_data.Week >= (week + delta), "WeekWeekdayAttraction_Lag1"] = np.nan
daily_data.loc[daily_data.Week >= (week + delta), "WeekWeekdayMeanUnitPrice_Lag1"] = np.nan
daily_data = daily_data.drop(["WeekWeekdayAttraction", "WeekWeekdayMeanUnitPrice"], axis=1)

daily_data.head()
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import shap

class CatHyperparameter:
    def __init__(self, loss="RMSE", metric="RMSE", iterations=1000, max_depth=4, l2_leaf_reg=3, seed=0):
      self.loss = [loss]
      self.metric = [metric]
      self.max_depth = [max_depth]
      self.l2_leaf_reg = [l2_leaf_reg]
      self.iterations = iterations
      self.seed = seed

class Catmodel:
    def __init__(self, name, params):
        self.name = name
        self.params = params
    def set_data(self, X, y, week):
        cat_features_idx = np.where(X.dtypes == object)[0]
        self.x_val = X.loc[X.Week >= week]
        self.y_val = y.loc[X.Week >= week]
        x_train = X.loc[X.Week < week]
        y_train = y.loc[X.Week < week]
        self.train_pool = Pool(x_train, y_train, cat_features=cat_features_idx)
        self.val_pool = Pool(self.x_val, self.y_val, cat_features=cat_features_idx)
    def prepare_model(self):
        self.model = CatBoostRegressor(loss_function=self.params.loss[0], random_seed=self.params.seed, logging_level='Silent', iterations=self.params.iterations, max_depth=self.params.max_depth[0], l2_leaf_reg=self.params.l2_leaf_reg[0], od_type='Iter', od_wait=40, train_dir=self.name, has_time=True)
    def learn(self, plot=False):
        self.prepare_model()
        self.model.fit(self.train_pool, eval_set=self.val_pool, plot=plot)
    def score(self):
        return self.model.score(self.val_pool)
    def show_importances(self, kind="bar"):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.val_pool)
        if kind=="bar":
            return shap.summary_plot(shap_values, self.x_val, plot_type="bar")
        return shap.summary_plot(shap_values, self.x_val)
    def show_val_results(self):
        preds = self.model.predict(self.x_val)
        error = np.abs(self.y_val - preds)
        print(f"Median absolute error: {np.median(error)}")

# 1. Define final features and target
drop_cols = ['InvoiceDate', 'StockCode', 'Quantity', 'Revenue', 'Year']
X = daily_data.drop(columns=drop_cols)
y = daily_data['Quantity']
X['ProductType'] = X['ProductType'].fillna(-1).astype(int).astype(str)

# 2. Initialize Hyperparameters and Model
params = CatHyperparameter(iterations=500, max_depth=6)
model = Catmodel(name="ProductSalesModel", params=params)

# 3. Set data and train
model.set_data(X, y, week=50)
model.learn(plot=True)

# 4. Display results
model.show_val_results()
print(f"Validation R2 Score: {model.score():.4f}")
# Re-run the importance visualization now that the method is restored
model.show_importances(kind='bar')
daily_data["TransactionsPerStockCode"] = daily_data.StockCode.map(
    data.loc[data.Week < week].groupby("StockCode").InvoiceNo.nunique())
daily_data.head()
daily_data["CustomersPerWeekday"] = daily_data.Month.map(
    data.loc[data.Week < week].groupby("Weekday").CustomerID.nunique())
# Fix: 'Date' is actually 'InvoiceDate' in the current daily_data
# We also drop 'StockCode' as it's a high-cardinality identifier
X = daily_data.drop(["Quantity", "Revenue", "InvoiceDate", "Year", "StockCode"], axis=1)
y = daily_data.Quantity

# Ensure ProductType is treated as a string for CatBoost categorical handling
if 'ProductType' in X.columns:
    X['ProductType'] = X['ProductType'].fillna(-1).astype(int).astype(str)

params = CatHyperparameter()
model = Catmodel("new_features_1", params)
model.set_data(X, y, week)
model.learn(plot=True)
model.show_importances(kind=None)
model.show_val_results();
# Fix: Get predictions directly since the current Catmodel class doesn't store a .results attribute
preds = model.model.predict(model.x_val)

# Calculate the mean absolute error on the original scale
# Note: We only use np.exp if the target 'Quantity' was log-transformed during training
# Based on the current training cell, Quantity is not log-transformed, so we use it directly:
mae = np.mean(np.abs(preds - model.y_val))
print(f"Mean Absolute Error: {mae}")
# Fix: Use the variables already calculated in the previous cell
# and remove np.exp since the target was not log-transformed.
# 'preds' and 'model.y_val' are available from the recent training run.

median_abs_error = np.median(np.abs(preds - model.y_val))
print(f"Median Absolute Error: {median_abs_error}")
# Fix: Use valid variables and remove np.exp since the target wasn't logged
# 'preds' was calculated in cell EY-sYxXf9YMF and 'model.y_val' is the actual target

median_absolute_error = np.median(np.abs(preds - model.y_val))
print(f"Median Absolute Error: {median_absolute_error}")
search = Hypertuner(model)
#search.learn()
model = search.retrain_catmodel()
print(model.score())
model.show_importances(kind=None)

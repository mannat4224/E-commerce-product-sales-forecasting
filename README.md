# E-commerce-product-sales-forecasting
To deploy your e-commerce sales forecasting model for a live exhibition like TechFest 2.0, you need to transition it from a static Jupyter Notebook into an interactive web application. The fastest, most professional way to do this using your existing Python skills is with Streamlit and Streamlit Community Cloud.

Here is the complete, step-by-step guide to deploying your CatBoost model to the web for free.

Phase 1: Export Your Trained Model
Right now, your CatBoost model only exists in the memory of your Jupyter Notebook. You need to save it to a file so a web server can load it.

Add this code to the very end of your ecommerce_salesforecast.ipynb notebook and run it:

Python
# Save the trained CatBoost model to a file
# CatBoost has a built-in save_model method which is highly optimized
model.save_model("catboost_sales_model.cbm")
print("Model saved successfully for deployment!")
Phase 2: Create the Web App Interface (app.py)
Create a new folder on your computer for your deployment files. Move your catboost_sales_model.cbm file into this new folder. Then, create a new Python script named app.py in the same folder and add the following code:

Python
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import datetime

# 1. Page Configuration
st.set_page_config(page_title="E-Commerce Sales Forecast", layout="centered")
st.title("🛒 Intelligent Product Sales Forecaster")
st.markdown("Enter product details and temporal features to predict future sales volume.")

# 2. Load the CatBoost Model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_sales_model.cbm")
    return model

model = load_model()

# 3. Build the User Interface (Sidebar Inputs)
with st.sidebar:
    st.header("Input Parameters")
    
    # Simulating the features your model expects
    # Adjust these to match the exact feature columns in your notebook
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    month = st.slider("Month", 1, 12, 6)
    is_weekend = st.radio("Is Weekend?", [0, 1])
    
    st.divider()
    st.header("Historical Data")
    lag_7 = st.number_input("Sales 7 Days Ago", min_value=0, value=50)
    lag_14 = st.number_input("Sales 14 Days Ago", min_value=0, value=45)
    rolling_mean = st.number_input("7-Day Rolling Average", min_value=0.0, value=48.5)

# 4. Prediction Logic
if st.button("Generate Forecast", type="primary", use_container_width=True):
    
    # Create the feature array in the exact order your CatBoost model expects
    input_data = pd.DataFrame([[
        day_of_week, quarter, month, 2026, 150, # Example Year and DayOfYear
        is_weekend, lag_7, lag_14, rolling_mean
    ]], columns=[
        'DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear', 
        'IsWeekend', 'Lag_7', 'Lag_14', 'Rolling_Mean_7'
    ])
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success("Analysis Complete!")
    st.metric(
        label="Predicted Sales Volume", 
        value=f"{int(max(0, prediction))} units" # Prevent negative sales
    )
(Note: Ensure the columns list in the input_data DataFrame exactly matches the features you trained your new_features_1 CatBoost model on).

Phase 3: Define Dependencies
The cloud server needs to know what libraries to install to make your Python code run. In the same folder, create a simple text file named requirements.txt and add these lines:

Plaintext
streamlit
pandas
catboost
Phase 4: Push to GitHub and Deploy
Now that your local files are ready, it is time to put them on the web.

Upload to GitHub:

Go to GitHub and create a new public repository.

Upload your three files: app.py, catboost_sales_model.cbm, and requirements.txt.

Deploy on Streamlit Community Cloud:

Go to share.streamlit.io and log in with your GitHub account.

Click Create app -> Yes, let's deploy!

Select your repository, set the branch to main, and the Main file path to app.py.

Click Deploy!

The server will take a few minutes to build the environment. Once it finishes, you will be given a live URL. Evaluators at the TechFest 2.0 exhibition will be able to scan a QR code or type that URL into their own phones to test your working machine learning pipeline in real-time.

Do you need help modifying the app.py script to include the AdditiveForceVisualizer so the web app also displays the Explainable AI charts?

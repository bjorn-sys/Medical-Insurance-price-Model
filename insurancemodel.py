import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")

# ------------------------------
# Load Data and Model
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open("insurancemodelf.pkl", "rb"))
    return model

df = load_data()
model = load_model()

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dataset Overview", "EDA", "Model Prediction"])

# ------------------------------
# Dataset Overview
# ------------------------------
if page == "Dataset Overview":
    st.title("📊 Insurance Dataset Overview")
    
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Shape of Dataset")
    st.write(df.shape)

    st.write("### Data Types")
    st.write(df.dtypes)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Statistical Summary")
    st.write(df.describe())

# ------------------------------
# EDA Page
# ------------------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    # Pie charts for categorical features
    st.subheader("Categorical Feature Distribution")
    features = ['sex', 'smoker', 'region']
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(features):
        x = df[col].value_counts()
        ax[i].pie(x.values, labels=x.index, autopct="%1.1f%%")
        ax[i].set_title(f"{col} Distribution")
    st.pyplot(fig)

    # Average charges
    st.subheader("Average Charges by Feature")
    features = ['sex', 'children', 'smoker', 'region']
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for i, col in enumerate(features):
        sns.barplot(x=df[col], y=df['charges'], ax=ax[i//2][i%2], color="orange")
        ax[i//2][i%2].set_title(f"Average Charges by {col}")
    st.pyplot(fig)

    # Scatter plots
    st.subheader("Scatter Plot of Charges vs Continuous Features")
    features = ['age', 'bmi']
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, col in enumerate(features):
        sns.scatterplot(data=df, x=col, y="charges", hue="smoker", ax=ax[i])
        ax[i].set_title(f"{col} vs Charges")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# ------------------------------
# Prediction Page
# ------------------------------
elif page == "Model Prediction":
    st.title("💰 Insurance Charges Prediction")

    st.write("Enter the details below to predict the insurance charges.")

    age = st.slider("Age", 18, 64, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])

    # Map inputs to numerical values
    sex = 1 if sex == "Female" else 0
    smoker = 1 if smoker == "Yes" else 0
    region_map = {"Northwest": 0, "Northeast": 1, "Southeast": 2, "Southwest": 3}
    region = region_map[region]

    if st.button("Predict Charges"):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"💵 Predicted Insurance Charge: ${prediction:,.2f}")

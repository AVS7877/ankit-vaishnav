# Predictive Analytics for Farmers' Market Optimization

## Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import flask
from flask import Flask, render_template, request

## Step 2: Data Collection (Indian Farmer Market Prices API / Web Scraping)
def fetch_market_data():
    url = "https://api.data.gov.in/resource/market_price"  # Placeholder API
    params = {"api-key": "YOUR_API_KEY", "format": "json"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json()["records"])
    return pd.DataFrame()

## Step 3: Data Preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors='coerce')
    df.dropna(inplace=True)
    return df

## Step 4: Train Machine Learning Model
def train_model(df):
    X = df[["arrival_date", "commodity", "state", "district"]]
    y = df["modal_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    joblib.dump(model, "market_price_predictor.pkl")
    return model

## Step 5: Web Dashboard using Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    model = joblib.load("market_price_predictor.pkl")
    price = model.predict([[data['date'], data['commodity'], data['state'], data['district']]])
    return f"Predicted Price: {price[0]}"

if __name__ == "__main__":
    df = fetch_market_data()
    df = preprocess_data(df)
    train_model(df)
    app.run(debug=True)
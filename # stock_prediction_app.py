# stock_prediction_app.py

import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Predefined list of companies
COMPANIES = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
    'Facebook': 'FB'
}

st.title('Stock Prediction App')

# Sidebar for user input
st.sidebar.header('User Input')
selected_company = st.sidebar.selectbox('Select a Company:', list(COMPANIES.keys()))
symbol = COMPANIES[selected_company]
years_to_predict = st.sidebar.slider('Select Number of Years for Prediction:', 1, 10, 5)

# Function to fetch historical stock data
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Fetch historical stock data
start_date = '2010-01-01'  # Adjust the start date as needed
end_date = '2023-01-01'  # Adjust the end date as needed
stock_data = get_stock_data(symbol, start_date, end_date)

# Display stock data
st.subheader('Stock Data')
st.write(stock_data)

# Function to train ARIMA model and make predictions
def train_arima_model(data):
    model = ARIMA(data['Close'], order=(5,1,0))  # You may need to adjust order based on data characteristics
    fitted_model = model.fit()
    return fitted_model

# Train ARIMA model
arima_model = train_arima_model(stock_data)

# Sidebar for forecasting
st.sidebar.header('Forecasting Parameters')
forecast_days = years_to_predict * 365

# Generate forecast
forecast = arima_model.get_forecast(steps=forecast_days)

# Display forecast
st.subheader('Stock Price Forecast')
fig, ax = plt.subplots()
ax.plot(stock_data.index, stock_data['Close'], label='Historical Data', color='blue')
ax.plot(forecast.index, forecast.predicted_mean, label='Forecast', color='red')
ax.fill_between(forecast.index, forecast.conf_int()['lower Close'], forecast.conf_int()['upper Close'], color='pink', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
st.pyplot(fig)

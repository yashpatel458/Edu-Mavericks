
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.express as px
import numpy as np

# Load the data from CSV files
df1 = pd.read_csv("content/labour-market-report-chart-1-Emplyment in Ontario june 23.csv")
df2 = pd.read_csv("content/labour-market-report-chart-2-Industries with highest and lowest emp change jun 23.csv")
df3 = pd.read_csv("content/labour-market-report-chart-3-employment change in ontario june 23.csv")
df5 = pd.read_csv("content/labour-market-report-chart-5-unemployment rates.csv")
df8 = pd.read_csv("content/labour-market-report-chart-8- change in wage rate and CPI june 23.csv")

# Convert 'Date' column to datetime type where applicable
df1['Date'] = pd.to_datetime(df1['Date'])
df5['Date'] = pd.to_datetime(df5['Date'], errors='coerce')
df8['Date'] = pd.to_datetime(df8['Date'])

# Clean and preprocess data for df8
df8['CPI Inflation'] = pd.to_numeric(df8['CPI Inflation'].str.rstrip('%'), errors='coerce')
df8['Wage Change'] = pd.to_numeric(df8['Wage Change'].str.rstrip('%'), errors='coerce')

# Remove NaN values
df5 = df5.dropna(subset=['Date'])

# Set 'Date' column as the index for time series analysis in df1
df1.set_index('Date', inplace=True)
df1.index = pd.date_range(start=df1.index[0], periods=len(df1), freq='M')

# Define a function for ARIMA model and forecast
def arima_model_forecast(data, order=(1, 1, 1), steps=6):
    model = sm.tsa.ARIMA(data, order=order)
    results = model.fit()
    forecast = results.forecast(steps=steps)
    return forecast

# Fit ARIMA model to the df1 data and forecast
forecast = arima_model_forecast(df1['Ontario employment (x 1,000), seasonally adjusted'])

# Function to create and display Plotly figures
def plot_interactive_line(df, x_column, y_columns, title, xaxis_title, yaxis_title):
    fig = px.line(df, x=x_column, y=y_columns, title=title, labels={x_column: xaxis_title, 'value': yaxis_title})
    return fig

# Generate interactive line plot for CPI Inflation vs. Wage Change
cpi_wage_fig = plot_interactive_line(df8, 'Date', ['CPI Inflation', 'Wage Change'], 'CPI Inflation vs. Wage Change',
                                     'Date', 'Percentage')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go

from datetime import datetime

df1 = pd.read_csv("content/labour-market-report-chart-1-Emplyment in Ontario june 23.csv")
df2 = pd.read_csv("content/labour-market-report-chart-2-Industries with highest and lowest emp change jun 23.csv")
df3 = pd.read_csv("content/labour-market-report-chart-3-employment change in ontario june 23.csv")
df5 = pd.read_csv("content/labour-market-report-chart-5-unemployment rates.csv")
df8 = pd.read_csv("content/labour-market-report-chart-8- change in wage rate and CPI june 23.csv")


df1.info()
df1.head()


# Create a DataFrame
df = pd.DataFrame(df1)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Create an interactive line plot using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Ontario employment (x 1,000), seasonally adjusted'],
                         mode='lines+markers', name='Ontario Employment (x 1,000)'))

fig.update_layout(title='Ontario Employment Trend',
                  xaxis_title='Date',
                  yaxis_title='Ontario employment (x 1,000)',
                  xaxis=dict(type='category'),
                  yaxis=dict(title=dict(text='Ontario employment (x 1,000)')),
                  hovermode='x',
                  template='plotly_white')

# Show the interactive graph
fig.show()


import statsmodels.api as sm
df = pd.DataFrame(df1)
# Convert 'Date' column to datetime type
# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Manually set the frequency to monthly (M)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='M')

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ontario employment (x 1,000), seasonally adjusted'], label='Actual Data', color='blue')
plt.xlabel('Date')
plt.ylabel('Ontario Employment (x 1,000)')
plt.title('Ontario Employment Time Series')
plt.legend()
plt.show()

# Seasonal Decomposition
decomposition = sm.tsa.seasonal_decompose(df['Ontario employment (x 1,000), seasonally adjusted'], model='additive')

# Trend, Seasonal, and Residual components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot components
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(df.index, df['Ontario employment (x 1,000), seasonally adjusted'], label='Actual Data', color='blue')
plt.legend()
plt.subplot(412)
plt.plot(df.index, trend, label='Trend', color='red')
plt.legend()
plt.subplot(413)
plt.plot(df.index, seasonal, label='Seasonal', color='green')
plt.legend()
plt.subplot(414)
plt.plot(df.index, residual, label='Residual', color='orange')
plt.legend()
plt.tight_layout()
plt.show()

df = pd.DataFrame(df1)
# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Manually set the frequency to monthly (M)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='M')

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ontario employment (x 1,000), seasonally adjusted'], label='Actual Data', color='blue')
plt.xlabel('Date')
plt.ylabel('Ontario Employment (x 1,000)')
plt.title('Ontario Employment Time Series')
plt.legend()
plt.show()

# Fit ARIMA model to the data
model = sm.tsa.ARIMA(df['Ontario employment (x 1,000), seasonally adjusted'], order=(1, 1, 1))
results = model.fit()

# Make predictions for future dates (e.g., next 6 months)
future_dates = pd.date_range(start=df.index[-1], periods=6, freq='M')
forecast = results.forecast(steps=6)

# Create a DataFrame for the predictions
predictions = pd.DataFrame({'Forecast': forecast}, index=future_dates)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Ontario employment (x 1,000), seasonally adjusted'], label='Actual Data', color='blue')
plt.plot(predictions.index, predictions['Forecast'], label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Ontario Employment (x 1,000)')
plt.title('Ontario Employment Forecast')
plt.legend()
plt.show()

# Display the forecasted values
print(predictions)

# Create a DataFrame
df = pd.DataFrame(df1)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Manually set the frequency to monthly (M)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='M')

# Fit ARIMA model to the data
model = sm.tsa.ARIMA(df['Ontario employment (x 1,000), seasonally adjusted'], order=(1, 1, 1))
results = model.fit()

# Make predictions for future dates (e.g., until the year 2025)
future_dates = pd.date_range(start=df.index[-1], periods=12, freq='M')
forecast = results.forecast(steps=36)

# Create a DataFrame for the predictions
predictions = pd.DataFrame({'Forecast': forecast}, index=future_dates)

# Filter data and predictions from 2018 onwards
start_date = '2018-01-01'
filtered_df = df[df.index >= start_date]
filtered_predictions = predictions[predictions.index >= start_date]

# Plot the data and predictions
plt.figure(figsize=(10, 6))
plt.plot(filtered_df.index, filtered_df['Ontario employment (x 1,000), seasonally adjusted'], label='Actual Data', color='blue')
plt.plot(filtered_predictions.index, filtered_predictions['Forecast'], label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Ontario Employment (x 1,000)')
plt.title('Ontario Employment Forecast (from 2018 onwards)')
plt.legend()
plt.show()

# Display the forecasted values until 2025
print(filtered_predictions)

df2.info()
df2.head()


# Create a scatter plot using plotly express
scatter_fig = px.scatter(df, x='Month', y=' Employment, Ontario (000)', color='SIC',
                         labels={' Employment, Ontario (000)': 'Employment (in thousands)'},
                        # category_orders={'Month': ['Jan', 'Feb', 'Mar','Apr']},
                         width=1200, height=700)

# Create a line plot using plotly express
line_fig = px.line(df, x='Month', y=' Employment, Ontario (000)', color='SIC',
                   labels={' Employment, Ontario (000)': 'Employment (in thousands)'},
                 # category_orders={'Month': ['Jan', 'Feb', 'Mar', 'Apr']},
                   width=1200, height=700)

# Combine both plots into one figure
combined_fig = scatter_fig.add_traces(line_fig.data)

# Show the combined interactive plot
combined_fig.show()


df = pd.DataFrame(df2)
# Remove specific SIC categories
excluded_sics = ['Total employed, all industries', 'Services-producing sector','Goods-producing sector']
df = df[~df['SIC'].isin(excluded_sics)]

# Calculate the trend (slope) for each SIC category
trends = df.groupby('SIC').apply(lambda group: np.polyfit(range(len(group)), group[' Employment, Ontario (000)'], 1)[0])

# Create a scatter plot using plotly express
fig = px.scatter(df, x='Month', y=' Employment, Ontario (000)', color='SIC',
                 labels={'Employment, Ontario (000)': 'Employment (in thousands)'},
                 category_orders={'Month': ['Jan', 'Feb', 'Mar']},
                 width=1200, height=1000)

# Add annotations for each SIC category to show trend direction
for sic, trend in trends.items():
    trend_direction = 'Upward Trend' if trend > 0 else 'Downward Trend'
    y_pos = df[df['SIC'] == sic][' Employment, Ontario (000)'].max() + 10
    fig.add_annotation(
        x='Feb', y=y_pos,
        text=trend_direction, showarrow=False
    )

# Show the interactive plot
fig.show()

df3.info()
df3.head()

df = pd.DataFrame(df3)
excluded_sics = ['Total, all occupations']
df = df[~df['Broad occupational category'].isin(excluded_sics)]
# Convert 'Employment, Ontario (000)' column to numeric
df[' Employment, Ontario (000)'] = pd.to_numeric(df[' Employment, Ontario (000)'], errors='coerce')

# Create the interactive bar chart using Plotly Express
fig = px.line(df, x='Month', y=' Employment, Ontario (000)', color='Broad occupational category',
             labels={'Employment, Ontario (000)': 'Employment (000)', 'Broad occupational category': 'Category'})

# Update the layout to add a title and rotate x-axis labels
fig.update_layout(title='Employment Change in Ontario by Month and Occupational Category',
                  xaxis_tickangle=-45)

# Show the interactive plot
fig.show()

df5.info()
df5.head()

# Create a DataFrame
df = pd.DataFrame(df5)

# Convert 'Date' column to datetime type and handle errors with 'coerce'
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with invalid dates (NaT)
df = df.dropna(subset=['Date'])

# Create an interactive line plot using plotly
fig = go.Figure()

# Add trace for Canada unemployment rate
fig.add_trace(go.Scatter(x=df['Date'], y=df['Unemployment_rate_Canada'], mode='lines', name='Canada'))

# Add trace for Ontario unemployment rate
fig.add_trace(go.Scatter(x=df['Date'], y=df['Unemployment_rate_Ontario'], mode='lines', name='Ontario'))

# Set layout for the graph
fig.update_layout(
    title='Unemployment Rates in Canada and Ontario',
    xaxis_title='Date',
    yaxis_title='Unemployment Rate (%)',
    xaxis=dict(showline=True, showgrid=False),
    yaxis=dict(showline=True, showgrid=False),
    hovermode='x',  # Enable hover interactions
    showlegend=True
)

# Show the interactive graph
fig.show()

df8.info()
df8.head()

df = pd.DataFrame(df8)

df['Date'] = pd.to_datetime(df['Date'])

# Remove the percentage symbol and convert 'CPI Inflation' and 'Wage Change' to numeric
df['CPI Inflation'] = pd.to_numeric(df['CPI Inflation'].str.rstrip('%'), errors='coerce')
df['Wage Change'] = pd.to_numeric(df['Wage Change'].str.rstrip('%'), errors='coerce')

# Create the interactive line plot using Plotly
fig = px.line(df, x='Date', y=['CPI Inflation', 'Wage Change'], title='CPI Inflation vs. Wage Change',
              labels={'value': 'Percentage'}, hover_name='Date', line_shape='linear')

# Show the interactive plot
fig.show()


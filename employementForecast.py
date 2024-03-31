
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Load and display an image in the sidebar
logo_path = 'logo.png' 
st.sidebar.image(logo_path, use_column_width=True)

# Set the title of the app
st.title('Insights into Ontario\'s Employment Landscape')

# Add Table of Contents in the sidebar
st.sidebar.header('Table of Contents')
st.sidebar.markdown("""
- [Data Overview](#data-overview)
- [Employment in Ontario](#employment-in-ontario)
- [Time Series Analysis](#time-series-analysis)
- [Long Term Forecast](#long-term-forecast)
- [Industry Employment Analysis](#industry-employment-analysis)
- [Employment Trend Analysis by Industry](#employment-trend-analysis-by-industry)
- [Occupational Category Employment Changes](#occupational-category-employment-changes)
- [Unemployment Rate Analysis](#unemployment-rate-analysis)
- [Wage Rate and CPI Analysis](#wage-rate-and-cpi-analysis)
""")

df1 = pd.read_csv("content/labour-market-report-chart-1-Emplyment in Ontario june 23.csv")
df2 = pd.read_csv("content/labour-market-report-chart-2-Industries with highest and lowest emp change jun 23.csv")
df3 = pd.read_csv("content/labour-market-report-chart-3-employment change in ontario june 23.csv")
df5 = pd.read_csv("content/labour-market-report-chart-5-unemployment rates.csv")
df8 = pd.read_csv("content/labour-market-report-chart-8- change in wage rate and CPI june 23.csv")



# center on Liberty Bell, add marker
m = folium.Map(location=[51.2538, -85.3232], zoom_start=5)


# call to render Folium map in Streamlit
st_data = st_folium(m, height=325, width=725)


st.subheader('Data Overview')
st.write('Employment in Ontario June 23 Dataset:')
st.dataframe(df1.head())
st.write('Industries with Highest and Lowest Employment Change Dataset:')
st.dataframe(df2.head())
st.write('Employment Change in Ontario Dataset:')
st.dataframe(df3.head())
st.write('Unemployment Rates Dataset:')
st.dataframe(df5.head())
st.write('Change in Wage Rate and CPI Dataset:')
st.dataframe(df8.head())

## Employment in Ontario

st.markdown("<h3 id='employment-in-ontario'>Employment in Ontario</h3>", unsafe_allow_html=True)
df1.info()
st.write(df1.describe())

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

st.plotly_chart(fig, use_container_width=True)

## Time Series Analysis
######################################################################
st.subheader('Time Series Analysis')
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
plt.tight_layout()
st.pyplot(plt)

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
st.pyplot(plt)

############################################
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
plt.tight_layout()
st.pyplot(plt)

# Display the forecasted values
st.write('Forecasted Values for the Next 6 Months:')
st.dataframe(predictions)

######################################################################
st.subheader('Long Term Forecast')
df = pd.DataFrame(df1)

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
plt.tight_layout()
st.pyplot(plt)

"""
The employment trend in Ontario, as depicted by the 'Actual Data' line, shows a significant decrease in employment around 2020, but it has since recovered and continued to grow. The forecast suggests that this growth in employment is expected to continue steadily into the future, with no significant downturns or upswings predicted. The stable 'Forecast' line indicates a positive outlook for the Ontario job market, assuming the models used for forecasting remain accurate and no unforeseen events disrupt the job market.

While the 'Forecast' line suggests a smooth upward trend in employment, a more cautious approach would consider the possibility of ongoing or new challenges that could disrupt this trend. Economic recovery from events like the pandemic is often uneven and can be subject to setbacks from factors like renewed public health crises, supply chain issues, inflation, or shifts in global markets. Therefore, without including comprehensive current affairs in the forecasting model, there is a risk that this optimistic outlook may not accurately reflect future realities, and the actual employment trajectory could be more volatile than projected.
"""


# Display the forecasted values until 2028
st.write('Forecasted Values until 2028:')
st.dataframe(filtered_predictions)


"""
Employment in Ontario increased in June by 55,800 (0.7%) to 7,951,300, after decreasing by 23,900 (−0.3%) in May. Provincial employment has been on an upward trend in recent months, with job gains totalling 236,400 since September 2022.

Employment in Canada increased in June by 59,900 (0.3%), after decreasing by 17,300 (−0.1%) in May. A total of 20,172,800 people were employed in Canada in June.
"""
######################################################################
st.subheader('Industry Employment Analysis')
df2.info()
st.write(df2.describe())

# Make sure the dataframe has the correct column names
# Let's trim any leading or trailing spaces in column names
df2.columns = df2.columns.str.strip()

# Create a scatter plot using plotly express
import plotly.express as px
scatter_fig = px.scatter(df2, x='Month', y='Employment, Ontario (000)', color='SIC',
                         labels={'Employment, Ontario (000)': 'Employment (in thousands)'},
                         width=1200, height=700)

# Check the column names and ensure they are correct
st.write("Columns in df2:", df2.columns)

# Create a line plot using plotly express
line_fig = px.line(df2, x='Month', y='Employment, Ontario (000)', color='SIC',
                   labels={'Employment, Ontario (000)': 'Employment (in thousands)'},
                   width=1200, height=700)

# Combine both plots into one figure
combined_fig = scatter_fig.add_traces(line_fig.data)

# Show the combined interactive plot
st.plotly_chart(combined_fig, use_container_width=True)

#####################################################################
st.subheader('Employment Trend Analysis by Industry')
df = pd.DataFrame(df2)
# Remove specific SIC categories
excluded_sics = ['Total employed, all industries', 'Services-producing sector', 'Goods-producing sector']
df = df[~df['SIC'].isin(excluded_sics)]

# Calculate the trend (slope) for each SIC category
trends = df.groupby('SIC').apply(lambda group: np.polyfit(range(len(group)), group['Employment, Ontario (000)'], 1)[0])

# Create a scatter plot using plotly express
fig = px.scatter(df, x='Month', y='Employment, Ontario (000)', color='SIC',
                 labels={'Employment, Ontario (000)': 'Employment (in thousands)'},
                 category_orders={'Month': ['Jan', 'Feb', 'Mar']},
                 width=1200, height=1000)

# Add annotations for each SIC category to show trend direction
for sic, trend in trends.items():
    trend_direction = 'Upward Trend' if trend > 0 else 'Downward Trend'
    y_pos = df[df['SIC'] == sic]['Employment, Ontario (000)'].max() + 10
    fig.add_annotation(
        x='Feb', y=y_pos,
        text=trend_direction, showarrow=False
    )
# Show the interactive plot
st.plotly_chart(fig, use_container_width=True)


"""
Ontario’s largest industry groups by employment in June were wholesale and retail trade (1,137,500 or 14.3% of total employment), health care and social assistance (966,900 or 12.2%), professional, manufacturing (819,900 or 10.3%), scientific and technical services (810,000 or 10.2%) and finance, insurance, real estate, rental and leasing (693,200 or 8.7%).

Ten of the sixteen major industry groups recorded job gains in June. Wholesale and retail trade (11,900 or 1.1%), transportation and warehousing (9,700 or 2.5%), manufacturing (7,300 or 0.9%) and professional, scientific and technical services (6,700 or 0.8%) led job gains.

Employment losses occurred in accommodation and food services (−2,700 or −0.6%), educational services (−2,300 or −0.4%), construction (−1,800 or −0.3%) and information, culture and recreation (−1,500 or −0.4%) in June.

Employment was unchanged in agriculture and utilities in June.
"""
#####################################################################
st.subheader('Occupational Category Employment Changes')
df3.info()
st.write(df3.describe())


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
st.plotly_chart(fig, use_container_width=True)

"""
Ontario’s largest occupational groups by employment in June were sales and service (1,733,500 or 21.5% of total employment), business, finance and administration (1,368,700 or 17.0%), trades, transport and equipment operators (1,167,700 or 14.5%), occupations in education, law and social, community and government services (895,300 or 11.1%) and management (823,700 or 10.2%).

Seven of the ten major occupational groups in Ontario had net employment gains in the first six months of 2023 when compared to the same period in 2022. Management occupations (74,000 or 10.0%) led job gains, followed by trades, transport and equipment operators and related occupations (72,900 or 7.0%), occupations in art, culture, recreation and sport (24,700 or 12.1%) and occupations in education, law, social, community and government services (22,900 or 2.6%).

Employment losses were recorded in occupations in manufacturing and utilities (−20,600 or −5.4%), natural resources, agriculture and related production occupations (−9,200 or −10.7%), and natural and applied sciences and related occupations (−8,300 or −1.1%).
"""
#####################################################################
st.subheader('Unemployment Rate Analysis')
df5.info()
st.write(df5.describe())


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
    hovermode='x',
    showlegend=True
)

# Show the interactive graph
st.plotly_chart(fig, use_container_width=True)

"""
Ontario’s unemployment rate increased to 5.7% in June from 5.5% in May, marking the second consecutive monthly increase after trending downward since November 2022.

June’s unemployment rate increased as employment gains were outpaced by gains in the labour force.

The Canadian unemployment rate rose to 5.4% in June from 5.2% in May
"""

#####################################################################
st.subheader('Wage Rate and CPI Analysis')

df8.info()
st.write(df8.describe())

df = pd.DataFrame(df8)

df['Date'] = pd.to_datetime(df['Date'])

# Remove the percentage symbol and convert 'CPI Inflation' and 'Wage Change' to numeric
df['CPI Inflation'] = pd.to_numeric(df['CPI Inflation'].str.rstrip('%'), errors='coerce')
df['Wage Change'] = pd.to_numeric(df['Wage Change'].str.rstrip('%'), errors='coerce')

# Create the interactive line plot using Plotly
fig = px.line(df, x='Date', y=['CPI Inflation', 'Wage Change'], title='CPI Inflation vs. Wage Change',
              labels={'value': 'Percentage'}, hover_name='Date', line_shape='linear')

# Show the interactive plot
st.plotly_chart(fig, use_container_width=True)


"""
The graph demonstrates periods where wage changes do not align with CPI inflation trends. Notably, around 2020, the CPI inflation rate peaks sharply, likely outpacing wage growth. This discrepancy could indicate a period where the purchasing power of wages decreased due to inflation rising faster than wages. The mismatch between wage growth and inflation can lead to reduced real income for individuals, making it harder for consumers to maintain their standard of living. The instability in the CPI line after 2020, with continued fluctuations, could suggest ongoing economic challenges or recovery efforts. The data implies that wages have not consistently kept pace with inflation, a situation that can contribute to broader economic stress for the working population.
"""

# Add a footer
footer_html = """
<div style='text-align: center;'>
    <p style='margin: 20px 0;'>
        Made with ❤️ by Param, Yash S, Yash P & Vraj
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
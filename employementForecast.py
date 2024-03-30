import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Function to load data
def load_data(file_name):
    data = pd.read_csv(f'content/{file_name}')
    date_column = 'Date' if 'Date' in data.columns else data.columns[0]  # Fallback to first column if 'Date' is not present

    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')  # Coerce errors to handle invalid date formats

    return data

# Define the datasets
datasets = {
    "Employment in Ontario": "labour-market-report-chart-1-Emplyment in Ontario june 23.csv",
    "Industries Employment Change": "labour-market-report-chart-2-Industries with highest and lowest emp change jun 23.csv",
    "Employment Change": "labour-market-report-chart-3-employment change in ontario june 23.csv",
    "Unemployment Rates": "labour-market-report-chart-5-unemployment rates.csv",
    "Wage Rate and CPI Change": "labour-market-report-chart-8- change in wage rate and CPI june 23.csv"
}

# Initialize the Streamlit app
st.title('Employment Data Visualization')

# Sidebar for dataset selection
selected_dataset = st.sidebar.selectbox('Select a dataset', list(datasets.keys()))

# Load and display the selected dataset
data_file = datasets[selected_dataset]
df = load_data(data_file)
if st.checkbox('Show raw data'):
    st.write(df)

# Plotting
st.sidebar.header("Plot Settings")
plot_type = st.sidebar.selectbox("Select plot type", ["Line Plot", "Bar Chart", "Area Chart"])

# Filter only numeric columns for plotting (excluding date column)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Configure plot settings based on the selected plot type
if plot_type == "Line Plot":
    if numeric_cols:
        fig = px.line(df, x=df.columns[0], y=numeric_cols, title=f'{selected_dataset} - Line Plot')
        st.plotly_chart(fig)
    else:
        st.warning('No numeric data to plot.')
elif plot_type == "Bar Chart":
    if numeric_cols:
        fig = px.bar(df, x=df.columns[0], y=numeric_cols, title=f'{selected_dataset} - Bar Chart')
        st.plotly_chart(fig)
    else:
        st.warning('No numeric data to plot.')
elif plot_type == "Area Chart":
    if numeric_cols:
        fig = px.area(df, x=df.columns[0], y=numeric_cols, title=f'{selected_dataset} - Area Chart')
        st.plotly_chart(fig)
    else:
        st.warning('No numeric data to plot.')

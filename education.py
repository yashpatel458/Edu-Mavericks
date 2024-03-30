import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and display an image in the sidebar
logo_path = 'logo.png' 
st.sidebar.image(logo_path, use_column_width=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('education_level_region.csv')
    df = df.melt(id_vars=['Geography', 'Educational attainment level'], var_name='Year', value_name='Percentage')
    df['Year'] = df['Year'].astype(int)  # Convert 'Year' to integer
    return df

df = load_data()

# App title
st.title("Education Level Data by Region in Canada")

# Sidebar for filtering
st.sidebar.header("Filter Data")
all_geography = st.sidebar.checkbox("Select All Regions", True)
if all_geography:
    selected_geography = df['Geography'].unique().tolist()
else:
    selected_geography = st.sidebar.multiselect("Select Regions", options=df['Geography'].unique())

filtered_data = df[df['Geography'].isin(selected_geography)]

# Checkbox for displaying raw data
if st.checkbox('Show raw data'):
    st.write(filtered_data)


# Trend analysis with slider
year_to_filter = st.slider('Select a year to filter the data', min_value=df['Year'].min(), max_value=df['Year'].max(), value=df['Year'].min())
filtered_year_data = filtered_data[filtered_data['Year'] == year_to_filter]
st.subheader(f"Trend Analysis for the year {year_to_filter}")
fig = px.bar(filtered_year_data, x='Geography', y='Percentage', color='Educational attainment level')
st.plotly_chart(fig)


# Define year range slider
year_range = st.slider('Select the year range:', df['Year'].min(), df['Year'].max(), (df['Year'].min(), df['Year'].max()))
# Filter data based on the selected year range
year_filtered_data = df[df['Year'].between(year_range[0], year_range[1])]

# Education Level Trends Over Time
st.subheader("Education Level Trends Over Time")
education_levels = year_filtered_data['Educational attainment level'].unique()
for level in education_levels:
    level_data = year_filtered_data[year_filtered_data['Educational attainment level'] == level]
    fig = px.line(level_data, x='Year', y='Percentage', color='Geography', title=f'Trend for {level}')
    st.plotly_chart(fig)

# Heatmap of Education Levels Across Regions and Years
st.subheader("Heatmap of Education Levels Across Regions and Years")
for level in education_levels:
    level_data = year_filtered_data[year_filtered_data['Educational attainment level'] == level]
    heatmap_data = level_data.pivot(index='Geography', columns='Year', values='Percentage')  # Corrected pivot usage
    st.write(f"Heatmap for {level}")
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', ax=ax)
    st.pyplot(fig)

# K-Means Clustering
if st.checkbox('Perform K-Means Clustering'):
    if not filtered_data.empty:
        st.subheader("K-Means Clustering")
        num_clusters = st.slider('Select Number of Clusters', 2, 10, 3)
        clustering_data = filtered_data[['Percentage']].dropna()
        if clustering_data.shape[0] > 1:  # Ensure there's enough data
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(clustering_data)
            filtered_data['Cluster'] = clusters

            # Enhance cluster visualization
            fig = px.scatter(filtered_data, x='Year', y='Percentage', color='Cluster', size='Percentage', title="K-Means Clustering Results", hover_data=['Educational attainment level'])
            fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)
        else:
            st.warning('Not enough data to perform clustering.')
    else:
        st.warning('No data available for clustering. Please adjust the filters.')

# Correlation Matrix
if st.checkbox('Show Correlation Matrix'):
    st.subheader("Correlation Matrix")
    numeric_data = filtered_data.select_dtypes(include=[np.number])  # Select only numerical columns for correlation
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    st.balloons()



# Add a footer
footer_html = """
<div style='text-align: center;'>
    <p style='margin: 20px 0;'>
        Made with ❤️ by Param, Yash S, Yash P & Vraj
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


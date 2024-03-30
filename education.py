import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.express as px


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('education_level_region.csv')

df = load_data()

st.title("Education Level Data by Region in Canada") 


# Display the dataframe
st.write("Education Level Data by Region:", df.head())

# Basic statistics
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Define a function to plot matplotlib graphs
def plot_state_data(state):
    state_data = df[df['Geography'] == state]
    below_secondary_mean = state_data[state_data['Educational attainment level'] == 'Below upper secondary 7'].iloc[:, 2:].mean(numeric_only=True)
    post_secondary_mean = state_data[state_data['Educational attainment level'] == 'Upper secondary and post-secondary non-tertiary'].iloc[:, 2:].mean(numeric_only=True)
    tertiary_mean = state_data[state_data['Educational attainment level'] == 'Tertiary education'].iloc[:, 2:].mean(numeric_only=True)

    plt.figure(figsize=(10, 6))
    plt.plot(below_secondary_mean.index, below_secondary_mean.values, label='Below Upper Secondary 7')
    plt.plot(post_secondary_mean.index, post_secondary_mean.values, label='Upper Secondary and Post-Secondary Non-Tertiary')
    plt.plot(tertiary_mean.index, tertiary_mean.values, label='Tertiary Education')
    plt.title(f"Educational Attainment in {state}")
    plt.xlabel("Year")
    plt.ylabel("Mean Percentage")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Plotting with Matplotlib
if st.checkbox("Show Matplotlib Graphs"):
    st.subheader("Matplotlib Visualizations")
    for state in df['Geography'].unique():
        plot_state_data(state)

# Run this function for Plotly interactive graphs
def interactiveGraphs(state):
    state_data = df[df['Geography'] == state]
    categories = ['Below upper secondary 7', 'Upper secondary and post-secondary non-tertiary', 'Tertiary education']
    years = df.columns[2:].tolist()  # Assuming year columns start from 3rd column

    data = {year: [state_data[state_data['Educational attainment level'] == category][year].values[0] if not state_data[state_data['Educational attainment level'] == category][year].empty else None for category in categories] for year in years}
    traces = []
    for year in years:
        if all(data[year]):
            trace = go.Scatter(x=categories, y=data[year], mode='lines+markers', name=year,
                               hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>')
            traces.append(trace)

    layout = go.Layout(
        title=f'Educational Attainment in {state}',
        xaxis=dict(title='Education Categories'),
        yaxis=dict(title='Percentage'),
        legend=dict(orientation='h', x=0.1, y=-0.2)
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(
        hovermode='closest',
        template='plotly_dark',  # Dark theme
        showlegend=True,
        legend=dict(title='Year', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig)

# Interactive graphs with Plotly
st.subheader("Interactive Plotly Visualizations")
state = st.selectbox("Select a State for Detailed View:", df['Geography'].unique())
interactiveGraphs(state)

# Correlation Matrix
st.subheader("Correlation Matrix")
correlation_matrix = df.corr(numeric_only=True)
st.write(correlation_matrix)
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig)


############################################################################################################

# Clustering of States based on Education Statistics
st.subheader("Clustering of States based on Education Statistics")

# Pivot the data to have states as rows and years as columns
pivot_data = df.pivot(index='Geography', columns='Educational attainment level')

# Drop multi-level index for simplicity
pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]

# Select only the columns containing years (2018, 2019, etc.)
year_columns = [col for col in pivot_data.columns if col.split('_')[0].isdigit()]

# Select data for clustering
X = pivot_data[year_columns]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you may need to experiment)
num_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original data
pivot_data['Cluster'] = cluster_labels

# Visualize the clusters with state labels
fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], color=cluster_labels, hover_name=pivot_data.index)
fig.update_layout(
    title='Clustering of States based on Education Statistics',
    xaxis_title='Principal Component 1',
    yaxis_title='Principal Component 2',
    showlegend=True,
    legend_title='Cluster'
)
st.plotly_chart(fig)
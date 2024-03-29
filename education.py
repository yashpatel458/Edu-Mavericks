import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('education_level_region.csv')

df = load_data()

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

# Plotly line graph for educational attainment trends
st.subheader("Educational Attainment Trends Over Time (Plotly)")
df_melted = df.melt(id_vars=['Geography', 'Educational attainment level'], var_name='Year', value_name='Percentage')
fig = px.line(df_melted, x='Year', y='Percentage', color='Geography', line_group='Educational attainment level',
              labels={'Year': 'Year', 'Percentage': 'Percentage', 'Geography': 'Geography'},
              title='Educational Attainment Trends Over Time')
st.plotly_chart(fig)

# Clustering of States based on Education Statistics
st.subheader("Clustering of States based on Education Statistics")
pivot_data = df.pivot(index='Geography', columns='Educational attainment level', values='2022')  # Example with one year
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_data)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
pivot_data['Cluster'] = cluster_labels
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=pivot_data['Cluster'], palette='viridis', ax=ax)
plt.title('Clustering of States based on Education Statistics')
st.pyplot(fig)

# Run this function for Plotly interactive graphs
def interactiveGraphs(state):
    state_data = df[df['Geography'] == state]
    categories = ['Below upper secondary 7', 'Upper secondary and post-secondary non-tertiary', 'Tertiary education']
    years = df.columns[2:].tolist()  # Assuming year columns start from 3rd column

    data = {year: [state_data[state_data['Educational attainment level'] == category][year].values[0] if not state_data[state_data['Educational attainment level'] == category][year].empty else None for category in categories] for year in years}
    traces = []
    for year in years:
        if all(data[year]):
            trace = go.Scatter(x=categories, y=data[year], mode='lines+markers', name=year)
            traces.append(trace)

    layout = go.Layout(
        title=f'Educational Attainment in {state}',
        xaxis=dict(title='Education Categories'),
        yaxis=dict(title='Percentage'),
        legend=dict(orientation='h', x=0.1, y=-0.2)
    )
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

if __name__ == '__main__':
    st.run()

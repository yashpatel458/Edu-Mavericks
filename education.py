import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv(r'/content/education_level_region.csv')
df.describe()
for state in df['Geography'].unique():
    state_data = df[df['Geography'] == state]
    below_secondary_mean = state_data[state_data['Educational attainment level'] == 'Below upper secondary 7'].iloc[:, 2:].mean(numeric_only=True).mean()
    post_secondary_mean = state_data[state_data['Educational attainment level'] == 'Upper secondary and post-secondary non-tertiary'].iloc[:, 2:].mean(numeric_only=True).mean()
    tertiary_mean = state_data[state_data['Educational attainment level'] == 'Tertiary education'].iloc[:, 2:].mean(numeric_only=True).mean()
    print(f"Mean for {state}:")
    print(f"Below Upper Secondary 7: {below_secondary_mean:.2f}%")
    print(f"Upper Secondary and Post-Secondary Non-Tertiary: {post_secondary_mean:.2f}%")
    print(f"Tertiary Education: {tertiary_mean:.2f}%")
    print()
states = df['Geography'].unique()
num_states = len(states)
num_cols = 3  # Define the number of columns
num_rows = -(-num_states // num_cols)  # Calculate the number of rows required
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
axs = axs.flatten()
handles, labels = None, None
for i, state in enumerate(states):
    state_data = df[df['Geography'] == state]
    below_secondary_mean = state_data[state_data['Educational attainment level'] == 'Below upper secondary 7'].iloc[:, 2:].mean(numeric_only=True)
    post_secondary_mean = state_data[state_data['Educational attainment level'] == 'Upper secondary and post-secondary non-tertiary'].iloc[:, 2:].mean(numeric_only=True)
    tertiary_mean = state_data[state_data['Educational attainment level'] == 'Tertiary education'].iloc[:, 2:].mean(numeric_only=True)
    axs[i].plot(below_secondary_mean.index, below_secondary_mean.values, label='Below Upper Secondary 7')
    axs[i].plot(post_secondary_mean.index, post_secondary_mean.values, label='Upper Secondary and Post-Secondary Non-Tertiary')
    axs[i].plot(tertiary_mean.index, tertiary_mean.values, label='Tertiary Education')
    axs[i].set_title(state)
    axs[i].set_xlabel("Year")
    axs[i].set_ylabel("Mean Percentage")
    axs[i].grid(True)
handles, labels = axs[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
if num_states < num_rows * num_cols:
    for j in range(num_states, num_rows * num_cols):
        fig.delaxes(axs[j])
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
num_states = len(df['Geography'].unique())
num_cols = 3  # Number of columns for subplots
num_rows = (num_states - 1) // num_cols + 1  # Calculate number of rows
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
if num_rows == 1:
    axs = [axs]
handles_dict = {}
for i, state in enumerate(df['Geography'].unique()):
    row = i // num_cols
    col = i % num_cols
    state_data = df[df['Geography'] == state]
    categories = ['Below upper secondary 7', 'Upper secondary and post-secondary non-tertiary', 'Tertiary education']
    years = ['2018', '2019', '2020', '2021', '2022']
    data = {year: [state_data[state_data['Educational attainment level'] == category][year].values.tolist()[0] if not state_data[state_data['Educational attainment level'] == category][year].empty else None for category in categories] for year in years}
    for year in years:
        if all(data[year]):
            line, = axs[row][col].plot(categories, data[year], marker='o', label=year)
            handles_dict[year] = line
    axs[row][col].set_title(f'{state}')
    axs[row][col].set_xlabel('Education Categories', fontsize=10)  # Adjust font size
    axs[row][col].set_ylabel('Percentage')
    axs[row][col].grid(True)
    axs[row][col].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.legend(handles=handles_dict.values(), labels=handles_dict.keys(), loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
def normalGraphs():
  for state in df['Geography'].unique():
      state_data = df[df['Geography'] == state]
      categories = ['Below upper secondary 7', 'Upper secondary and post-secondary non-tertiary', 'Tertiary education']
      years = ['2018', '2019', '2020', '2021', '2022']
      data = {year: [state_data[state_data['Educational attainment level'] == category][year].values.tolist()[0] if not state_data[state_data['Educational attainment level'] == category][year].empty else None for category in categories] for year in years}
      plt.figure(figsize=(10, 6))
      for year in years:
          if all(data[year]):
              plt.plot(categories, data[year], marker='o', label=year)
      plt.title(f'Educational Attainment in {state}')
      plt.xlabel('Education Categories')
      plt.ylabel('Percentage')
      plt.legend()
      plt.grid(True)
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
import plotly.graph_objs as go
def interactiveGraphs():
  for state in df['Geography'].unique():
      state_data = df[df['Geography'] == state]
      categories = ['Below upper secondary 7', 'Upper secondary and post-secondary non-tertiary', 'Tertiary education']
      years = ['2018', '2019', '2020', '2021', '2022']
      data = {year: [state_data[state_data['Educational attainment level'] == category][year].values.tolist()[0] if not state_data[state_data['Educational attainment level'] == category][year].empty else None for category in categories] for year in years}
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
      fig.show()
interactiveGraphs()
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
import plotly.express as px
df_transposed = df.T.reset_index()
df_melted = df_transposed.melt(id_vars=['index'], var_name='Geography', value_name='Percentage')
fig = px.line(df_melted, x='index', y='Percentage', color='Geography',
              labels={'index': 'Year', 'Percentage': 'Percentage', 'Geography': 'Geography'},
              title='Educational Attainment Trends Over Time')
fig.update_traces(hoverinfo='text+name', mode='lines+markers')
fig.update_layout(hovermode='x')
fig.show()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
pivot_data = df.pivot(index='Geography', columns='Educational attainment level')
pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]
year_columns = [col for col in pivot_data.columns if col.split('_')[0].isdigit()]
X = pivot_data[year_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters,n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
pivot_data['Cluster'] = cluster_labels
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering of States based on Education Statistics')
for i, state in enumerate(pivot_data.index):
    plt.annotate(state, (X_scaled[i, 0], X_scaled[i, 1]))
plt.show()
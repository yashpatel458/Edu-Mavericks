import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load data
data = pd.read_csv('wage_rates_by_education.csv')

# Load and display an image in the sidebar
logo_path = 'logo.png' 
st.sidebar.image(logo_path, use_column_width=True)

# Add Table of Contents in the sidebar
st.sidebar.header('Table of Contents')
st.sidebar.markdown("""
  - [Column names](#column-names)
  - [Sum of wages by education level](#sum-of-wages-by-education-level)
  - [Mean wages by education level](#mean-wages-by-education-level)
  - [Bar plot of mean wages by education level](#bar-plot-of-mean-wages-by-education-level)
  - [Density plot of wages by type of work for Male](#density-plot-of-wages-by-type-of-work-for-male)
  - [Density plot of wages by type of work for Female](#density-plot-of-wages-by-type-of-work-for-female)
  - [Density plot of wages by type of work](#density-plot-of-wages-by-type-of-work)
  - [Distribute Wages data into all available classes](#distribute-wages-data-into-all-available-classes)
  - [Bar plot of wages by age group](#bar-plot-of-wages-by-age-group)
  - [Correlation matrix](#correlation-matrix)
  - [Bar plot of mean wages by education level (from pivot table)](#bar-plot-of-mean-wages-by-education-level-from-pivot-table)
""")



st.title("Analyzing Wage Disparities by Education Level in Canada") 


# Display the raw data
st.write(data)

# Display column names
st.subheader('Column names')
st.write(data.columns)

# Grouping and summarizing data
grouped_data = data.groupby('Education level')[['  Male', '  Female', 'Both Sexes']]
sum_data = grouped_data.sum()
mean_data = grouped_data.mean()

# Display data summaries
st.subheader('Sum of wages by education level')
st.write(sum_data)

st.subheader('Mean wages by education level')
st.write(mean_data)

# Plotting
st.subheader('Bar plot of mean wages by education level')
fig, ax = plt.subplots()
sns.barplot(x='Both Sexes', y='Education level', color='blue', data=data, ax=ax)
st.pyplot(fig)

st.subheader('Density plot of wages by type of work for Male')
sns.displot(data=data, x='  Male', hue='Type of work', kind='kde')
st.pyplot()

st.subheader('Density plot of wages by type of work For Female')
sns.displot(data=data, x='  Female', hue='Type of work', kind='kde')
st.pyplot()

st.subheader('Density plot of wages by type of work')
sns.displot(data=data, x='Both Sexes', hue='Type of work', kind='kde')
st.pyplot()

st.subheader('Distribute Wages data into all available classes')
sns.displot(data=data, x='Both Sexes', hue= 'Wages', kind = 'ecdf')
st.pyplot()

sns.displot(data=data, x='  Male', hue= 'Wages', kind = 'ecdf')
st.pyplot()

sns.displot(data=data, x='  Female', hue= 'Wages', kind = 'ecdf')
st.pyplot()



st.subheader('Bar plot of wages by age group')
sns.barplot(x='  Male', y='Age group', data=data,palette='viridis')
st.pyplot()


sns.barplot(x='  Female', y='Age group', data=data,palette='viridis')
st.pyplot()

sns.barplot(x='Both Sexes', y='Age group', data=data,palette='viridis')
st.pyplot()

# Correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

st.subheader('Correlation matrix')
st.write(corr_matrix)

# Pivot tables
piv = pd.pivot_table(data, values=['Both Sexes'], index=['Education level'], columns=['Wages'],
                     aggfunc={'Both Sexes': [max, np.mean]})
pivM = pd.pivot_table(data, values=['  Male'], index=['Education level'], columns=['Wages'],
                      aggfunc={'  Male': [max, np.mean]})
pivF = pd.pivot_table(data, values=['  Female'], index=['Education level'], columns=['Wages'],
                      aggfunc={'  Female': [max, np.mean]})
# Plotting pivot table
st.subheader('Bar plot of mean wages by education level (from pivot table)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=piv.reset_index(), x='Education level', y=piv.columns[0], ax=ax)
plt.title('Mean Wages by Education Level')
plt.xlabel('Education Level')
plt.xticks(rotation=90)
plt.ylabel('Mean Wages (in thousands)')
st.pyplot(fig)

"""
The bar chart demonstrates a positive correlation between the level of education and mean wages; generally, individuals with higher education levels earn higher wages. This trend is consistent with the economic theory that investment in human capital, such as education, enhances productivity and, consequently, earnings. There are, however, some nuances, such as 'Trade Certificate or Diploma' earning similar to 'Community College/CEGEP,' which might be due to specific demand for skilled trades. The 'Total all education levels' bar provides a benchmark mean wage against which the wages for specific education levels can be compared.
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





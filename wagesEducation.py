import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and display an image in the sidebar
logo_path = 'logo.png' 
st.sidebar.image(logo_path, use_column_width=True)

def load_data():
    data = pd.read_csv('wage_rates_by_education.csv')
    return data

def main():
    st.title('Wages by Education Analysis')

    data = load_data()

    # Use sidebar for filtering options
    education_levels = data['Education level'].unique()
    selected_education = st.sidebar.multiselect('Select Education Levels', education_levels, default=education_levels)

    filtered_data = data[data['Education level'].isin(selected_education)]

    if st.checkbox('Show raw data'):
        st.write(filtered_data)

    st.write('## Data Summary')
    st.write(filtered_data.describe())

    st.write('## Wages by Education Level')
    edu_grouped = filtered_data.groupby('Education level')[['Both Sexes']].mean()
    st.bar_chart(edu_grouped)

    st.write('## Wages by Type of Work')
    work_type_grouped = filtered_data.groupby('Type of work')[['Both Sexes']].mean()
    st.bar_chart(work_type_grouped)

    st.write('## Gender Wage Distribution')
    if st.checkbox('Show gender distribution plots'):
        fig, ax = plt.subplots()
        sns.kdeplot(data=filtered_data, x='  Male', shade=True, label='Male')
        sns.kdeplot(data=filtered_data, x='  Female', shade=True, label='Female')
        plt.title('Distribution of Wages by Gender')
        plt.legend()
        st.pyplot(fig)

    # Wage Comparison by Gender and Education Level
    st.write('## Wage Comparison by Gender and Education Level')
    gender_education_grouped = filtered_data.groupby('Education level')[['  Male', '  Female']].mean()
    st.bar_chart(gender_education_grouped)

if __name__ == "__main__":
    main()


# Add a footer
footer_html = """
<div style='text-align: center;'>
    <p style='margin: 20px 0;'>
        Made with ❤️ by Param, Yash S, Yash P & Vraj
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
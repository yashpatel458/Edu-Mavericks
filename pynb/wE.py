# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv(r'/content/wage_rates_by_education.csv')

# %%
"""
# Dataset
"""

# %%
data

# %%
data.columns

# %%
data.groupby('Education level') ['Both Sexes'] .sum()

# %%
"""
### Calculate the average weekly wage of Canadians with different education levels

"""

# %%
data.groupby('Education level') ['Both Sexes'] .mean()

# %%
data.groupby('Education level') [['  Male','  Female', 'Both Sexes']].mean()

# %%
sns.barplot(x='Both Sexes', y='Education level', color='blue', data=data)

# %%
"""
### Distribute the average weekly wage data by Type of work class and present it in a chart.
"""

# %%
data.groupby('Type of work') [['  Male','  Female', 'Both Sexes']] .mean()

# %%
sns.displot(data=data, x='  Male', hue= 'Type of work', kind = 'kde')

# %%
sns.displot(data=data, x='  Female', hue= 'Type of work', kind = 'kde')

# %%
sns.displot(data=data, x='Both Sexes', hue= 'Type of work', kind = 'kde')

# %%
"""
### Distribute Wages data into all avaliable classes.

"""

# %%
data.groupby('Wages') [['  Male','  Female', 'Both Sexes']].mean()

# %%
sns.displot(data=data, x='Both Sexes', hue= 'Wages', kind = 'ecdf')

# %%
sns.displot(data=data, x='  Male', hue= 'Wages', kind = 'ecdf')

# %%
sns.displot(data=data, x='  Female', hue= 'Wages', kind = 'ecdf')

# %%
"""
### Distribute average weekly wage by different classes of age group.

"""

# %%
data.groupby('Age group')  [['  Male','  Female', 'Both Sexes']].mean()

# %%
sns.barplot(x='  Male', y='Age group', data=data)

# %%
sns.barplot(x='  Female', y='Age group', data=data)

# %%
sns.barplot(x='Both Sexes', y='Age group', data=data)

# %%
"""
###  Find out the correlation between Year, Both Sexes, Male and Female
"""

# %%
data.corr()

# %%
"""
#### Find unique and nunique values in Education level column.
"""

# %%
data['Education level'].unique()

# %%
data['Education level'].nunique()

# %%
"""
### Apply iloc on two different tables and then merge the tables together by applying the appropriate function.
"""

# %%
data1 = data.iloc[ : , 3:5]

# %%
data2 = data.iloc[ : , 7:9 ]

# %%
pd.concat([data1,data2], axis = 1)

# %%
"""
### Prepare the pivot table for Wages, Education level and Age group cloumns.

"""

# %%
pd.pivot_table(data, values= 'Both Sexes', index= ['Wages', 'Education level'], columns= ['Age group'], aggfunc=np.mean)

# %%
pivm=pd.pivot_table(data, values= ['  Male','  Female'], index= ['Wages', 'Education level'], columns= ['Age group'], aggfunc=np.mean)

# %%
"""
### Prepare the pivot table and illustrate the different wages by Education level.
"""

# %%
pd.pivot_table(data, values= ['Both Sexes'], index= ['Education level'], columns= ['Wages'], aggfunc=np.mean)

# %%
pd.pivot_table(data, values= ['  Male','  Female'], index= ['Education level'], columns= ['Wages'], aggfunc=np.mean)

# %%

### Prepare the pivot table and illustrate the maximum and average wages by Education level.


# %%
piv=pd.pivot_table(data, values= ['Both Sexes'], index= ['Education level'], columns= ['Wages'], aggfunc={ 'Both Sexes':np.mean,
                       'Both Sexes':[ max,np.mean]})
pivM=pd.pivot_table(data, values= ['  Male'], index= ['Education level'], columns= ['Wages'], aggfunc={ '  Male':np.mean,
                       '  Male':[ max,np.mean]})
pivF=pd.pivot_table(data, values= ['  Female'], index= ['Education level'], columns= ['Wages'], aggfunc={ '  Female':np.mean,
                       '  Female':[ max,np.mean]})

# %%
pivM.info()


# %%
plt.figure(figsize=(10, 6))
sns.barplot(data=piv.reset_index(), x='Education level', y=piv.columns[0])
plt.title('Mean Wages by Education Level')
plt.xlabel('Education Level')
plt.xticks(rotation=90)
plt.ylabel('Mean Wages (in thousands)')
#plt.legend(title='Wages')
plt.show()
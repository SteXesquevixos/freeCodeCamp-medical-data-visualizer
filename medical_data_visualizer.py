import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import file
df = pd.read_csv('medical_examination.csv')
pd.set_option('display.max_columns', None)

# Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by
# dividing their weight in kilograms by the square of their height in meters. If that value is > 25 then the person
# is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.

BMI = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = pd.DataFrame(map(lambda i: 1 if i >= 25 else 0, BMI))

# Normalize the data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, make the
# value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = pd.DataFrame(map(lambda i: 0 if i == 1 else 1, df['cholesterol']))
df['gluc'] = pd.DataFrame(map(lambda i: 0 if i == 1 else 1, df['gluc']))


'''Create a chart similar to examples/Figure_1.png, where we show the counts of good and bad outcomes for the 
    cholesterol, gluc, alco, active, and smoke variables for patients with cardio=1 and cardio=0 in different panels.'''

def draw_cat_plot():

    # Convert the data into long format and create a chart that shows the value counts of the categorical features
    # using seaborn's catplot(). The dataset should be split by 'Cardio' so there is one chart for each cardio value.
    # The chart should look like examples/Figure_1.png.

    df_cat = pd.DataFrame({'cardio': df['cardio'],
                           'cholesterol': df['cholesterol'],
                           'gluc': df['gluc'],
                           'alco': df['alco'],
                           'active': df['active'],
                           'smoke': df['smoke'],
                           'overweight': df['overweight']},
                          columns=['cardio', 'cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight']
                          )

    df_cat = pd.melt(df_cat, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight'])

    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).agg(
        count_col=pd.NamedAgg(column='value', aggfunc='count'))

    df_cat = df_cat.rename(columns={'count_col': 'total'})

    ax = sns.catplot(x="variable",
                     y="total",
                     col="cardio",
                     hue="value",
                     kind="bar",
                     data=df_cat).figure

    ax.savefig('catplot.png')
    return ax

'''Create a correlation matrix using the dataset. Plot the correlation matrix using seaborn's heatmap(). Mask the upper 
    triangle. The chart should look like examples/Figure_2.png.'''

def draw_heat_map():

    # Clean the data. Filter out the following patient segments that represent incorrect data:
    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    # height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    # height is more than the 97.5th percentile
    # height is more than the 97.5th percentile
    # weight is more than the 97.5th percentile

    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr)).astype(bool)

    fig, ax = plt.subplots()
    ax = sns.heatmap(data=corr, annot=True, fmt='0.1f', mask=mask)

    fig.savefig('heatmap.png')
    return fig
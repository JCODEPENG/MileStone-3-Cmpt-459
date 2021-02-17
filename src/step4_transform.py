# CMPT459 Data Mining
# Spring 2021 Milestone 1
# Purpose: to transform the information for cases from the US from the country level to the state level
# Lucia Schmidt & Joshua Peng

import pandas as pd

def transform_location_data():
    df = pd.read_csv("../data/location.csv")
    df = df[df['Country_Region'] == 'US']
    df = df.groupby(['Province_State']).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Active': 'sum', \
                                            'Incidence_Rate': 'sum', 'Case-Fatality_Ratio': 'mean'})
    print(df)
    return df


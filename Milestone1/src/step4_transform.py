# CMPT459 Data Mining
# Spring 2021 Milestone 1
# Purpose: to transform the information for cases from the US from the country level to the state level
# Lucia Schmidt & Joshua Peng

import pandas as pd

def calculate_percent_confirmed(df):
    confirmed_total = df['Confirmed'].sum()
    df['percent_confirmed'] = df['Confirmed'] / confirmed_total
    return df

def calculate_case_fatality_ratio(df):
    '''
    case fatality ratio for states is calculated as the weighted sum based on
    the number of confirmed cases per city. The ratio is multiplied by the number
    of confirmed cases here and is later divided by the total number of cases per
    state after the GroupBy aggregation is done
    '''
    df['case_fat_ratio_multiplied'] = df['Case-Fatality_Ratio'] * df['Confirmed']
    return df 

def group_by_us_state(df):
    df = df.groupby(['Province_State'], as_index=False).agg({'Country_Region': lambda x: pd.Series.mode(x)[0], \
                                            'Last_Update': lambda x: pd.Series.mode(x)[0], \
                                            'Lat': 'mean','Long_': 'mean', 'Confirmed': 'sum', 'Deaths': 'sum', \
                                            'Recovered': 'sum', 'Active': 'sum', \
                                            'Combined_Key': lambda x: ','.join(pd.Series.mode(x)[0].split(',')[1:]).strip(), \
                                            'Incidence_Rate': 'mean', 'percent_confirmed': 'sum', \
                                            'case_fat_ratio_multiplied': 'sum'})
    return df

def allocate_recovered(df):
    '''
    The recovered column for US rows is empty except for one row that contains the aggregate number.
    This function allocates the recovered numbers to the individual states.
    '''
    recovered_total = df['Recovered'].sum()
    df['recovered_allocated'] = round(df['percent_confirmed'] * recovered_total)
    df = df[df['Province_State'] != 'Recovered']
    assert(recovered_total - df['recovered_allocated'].sum() < 2)
    return df

def concat_us_to_other_data(df_all, df_us):
    df_all = df_all[df_all['Country_Region'] != 'US']
    df_us = df_us[['Province_State', 'Country_Region', 'Last_Update', 'Lat', 'Long_', 'Confirmed', 'Deaths', 'recovered_allocated', \
            'Active', 'Combined_Key', 'Incidence_Rate', 'case_fatality_ratio_allocated']]
    df_us = df_us.rename(columns={'recovered_allocated': 'Recovered', 'case_fatality_ratio_allocated': 'Case-Fatality_Ratio'})
    df = pd.concat([df_all, df_us], ignore_index=True)
    return df

def transform_location_data():
    df_all = pd.read_csv("../data/location.csv")
    df = df_all[df_all['Country_Region'] == 'US']
    df = calculate_percent_confirmed(df)
    df = calculate_case_fatality_ratio(df)
    df = group_by_us_state(df)
    df = allocate_recovered(df)
    df['case_fatality_ratio_allocated'] = df['case_fat_ratio_multiplied'] / df['Confirmed']
    df = concat_us_to_other_data(df_all, df)
    # print(df)
    df.to_csv("../results/location_transformed.csv", index=False)



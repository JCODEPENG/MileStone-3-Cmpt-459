# CMPT459 Data Mining
# Spring 2021 Milestone 1
# Purpose: to join the cases dataframes with the location dataframe 
# Lucia Schmidt & Joshua Peng

import pandas as pd
import numpy as np

# This dictionary represents the countries that are labelled differently between the *cases.csv and location.csv
# The keys represent the countries in the *cases.csv file. The values are what is written in the location.csv
mismatched_countries = {'Reunion': 'Reunion, France', 'United States': 'US', 'South Korea': 'Korea, South', \
                        'Puerto Rico': 'Puerto Rico, US', 'Czech Republic': 'Czechia', 'Taiwan': 'Taiwan*', \
                        'Democratic Republic of the Congo': 'Congo (Kinshasa)', \
                        'Republic of Congo': 'Congo (Brazzaville)'}

# This dictionary represents where the *cases.csv and location.csv are inconsistent in what is a country vs. a province
# The keys represent what is noted as a country in the *cases.csv file. The values are what is written in the location.csv
mismatched_province_and_country = {'Reunion': 'Reunion, France', 'Puerto Rico': 'Puerto Rico, US'}

def get_country_only(country):
    if country in mismatched_countries:
        country_key = mismatched_countries[country]
    else:
        country_key = country
    return country_key

def get_province_and_country(province, country):
    if country in mismatched_province_and_country:
        combined_key = mismatched_province_and_country[country]
    else:
        combined_key = province + ', ' + get_country_only(country)
    return combined_key


def generate_combined_key(row, province_state_set):
    province = row['province_filled'].strip()
    country = row['country_filled'].strip()
    if province in province_state_set:
        join_key = get_province_and_country(province, country)
    else:
        join_key = get_country_only(country)
    return str(join_key)

def fill_missing_locs(row, df_location_agg):
    if (str(row['Confirmed']) == 'nan'):
        country = row['country_filled'].strip()
        country = get_country_only(country)
        location_row = df_location_agg[df_location_agg['Country_Region'] == country]
        row['Confirmed'] = int(location_row['Confirmed'])
        row['Deaths'] = int(location_row['Deaths'])
        row['Recovered'] = int(location_row['Recovered'])
        row['Active'] = int(location_row['Active'])
        row['Incidence_Rate'] = float(location_row['Incidence_Rate'])
        row['Case-Fatality_Ratio'] = float(location_row['Case-Fatality_Ratio'])
    return row

def join_cases_with_locations():
    df_location = pd.read_csv("../results/location_transformed.csv")
    df_location_agg = df_location.groupby('Country_Region',as_index=False).agg(\
        {'Confirmed': 'mean', 'Deaths': 'mean', 'Recovered': 'mean', 'Active': 'mean', \
         'Incidence_Rate': 'mean', 'Case-Fatality_Ratio': 'mean'})
    df_train_cases = pd.read_csv("../results/cleaned_outliers_train.csv")
    df_test_cases = pd.read_csv('../results/cleaned_cases_test.csv')

    province_state_set = set(df_location['Province_State'])
    province_state_set.remove(np.nan)

    # join training data to location data 
    df_train_cases['Combined_Key'] = df_train_cases.apply(generate_combined_key, axis=1, province_state_set=province_state_set)
    df_train_cases = df_train_cases[['Combined_Key', 'age_filled', 'filled_sex', 'province_filled', 'country_filled', 'latitude', 'longitude', 'new_date_confirmation', 'additional_information', 'source', 'outcome']]
    df_train_cases = pd.merge(df_train_cases, df_location, how='left', on=['Combined_Key', 'Combined_Key'])
    df_train_cases = df_train_cases.apply(fill_missing_locs, axis=1, df_location_agg=df_location_agg)
    df_train_cases.to_csv("../results/cases_train_processed.csv", index=False)

    # join testing data to location data 
    df_test_cases['Combined_Key'] = df_test_cases.apply(generate_combined_key, axis=1, province_state_set=province_state_set)
    df_test_cases = df_test_cases[['Combined_Key', 'age_filled', 'filled_sex', 'province_filled', 'country_filled', 'latitude', 'longitude', 'new_date_confirmation', 'additional_information', 'source', 'outcome']]
    df_test_cases = pd.merge(df_test_cases, df_location, how='left', on=['Combined_Key', 'Combined_Key'])
    df_test_cases = df_test_cases.apply(fill_missing_locs, axis=1, df_location_agg=df_location_agg)
    df_test_cases.to_csv("../results/cases_test_processed.csv", index=False)
    



import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
df = pd.read_csv('./cases_train.csv')
pd.options.mode.chained_assignment = None  # default='warn'

def find_gender_outcome(df):
    majorities = {}
    male_rows = df[df['sex'] == 'male']
    female_rows = df[df['sex'] =='female']

    # get unique outcomes for male and female
    male_outcome_vals, male_count = np.unique(male_rows['outcome'], return_counts=True)
    female_outcome_vals, female_count = np.unique(female_rows['outcome'], return_counts=True)

    # check which gender is more prevalent for each outcome
    for i in range(0,len(male_outcome_vals)):
        male_outcome_percentage = male_count[i] / len(male_rows['sex'])
        female_outcome_percentage = female_count[i]/len(female_rows['sex'])
        if male_outcome_percentage > female_outcome_percentage:
            majorities[male_outcome_vals[i]] = 'male'
        else:
            majorities[female_outcome_vals[i]] = 'female'

    gender_na = df[df['sex'].isna()]

    def fill_empty(row):
        return majorities[row['outcome']]

    gender_na['filled_sex'] = gender_na.apply(fill_empty, axis=1)
    df['filled_sex'] = df['sex']
    df.update(gender_na)
    return df

def process_age(df):
    # Replace odd dates (ie. May-14) in ages with nan values
    def clean_month_format(row):
        months = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'}
        age = str(row['age'])
        if '-' in age:
            if age.split('-')[1] in months:
                return np.nan
            else:
                return row['age']
        return row['age']
    df['age_parse_months'] = df['age']
    df['age_parse_months'] = df.apply(clean_month_format, axis=1)


    #Clear odd formats for ages
    df_clean = df.dropna(subset=['age_parse_months'])

    def clean_age_format(row):
        age = row['age_parse_months']
        if '-' in age:
            age_range = age.split('-')
            if age_range[1] != '' and age_range[1].isnumeric():
                return round((int(age_range[0]) + int(age_range[1]))/2)
            else:
                return int(age_range[0])
        elif '+' in age:
            return int(age[:-1])
        elif ' ' in age:
            age_range = age.split(' ')
            if age_range[1] != '' and age_range[1].isalpha():
                return round(int(age_range[0])*0.10)
        else:
            return int(round(float(age)))

    df_clean['age_formatted'] = df_clean.apply(clean_age_format, axis=1)
    df_clean = df_clean.dropna(subset=['age_formatted'])

    #Get general median for if location is not provided
    median_to_fill = df_clean['age_formatted'].median()
    df['age_formatted'] = df['age_parse_months']
    df.update(df_clean)
    #Partition average age for each country and province and store them as dict
    map_df = df.dropna(subset=['province','country'])
    province_ages = {}
    province = np.unique(map_df['province'])
    for name in province:
        if name != 'nan':
            ds = map_df[map_df['province'] == name]
            new_ds = ds['age_formatted'].dropna()
            if len(new_ds) > 0:
                province_ages[name] = np.sum(new_ds)//len(new_ds)
            else:
                province_ages[name] = median_to_fill

    country_ages = {}
    country = np.unique(map_df['country'])
    for name in country:
        if name != 'nan':
            ds = map_df[map_df['country'] == name]
            new_ds = ds['age_formatted'].dropna()
            if len(new_ds) > 0:

                country_ages[name] = np.sum(new_ds)//len(new_ds)
            else:
                country_ages[name] = median_to_fill


    #Using dict, fill age based on the average ages from province, countries
    def match_country(row, provinces,countries, avg):
        if row['province'] in provinces:
            return provinces[row['province']]
        elif row['country'] in countries:
            return countries[row['country']]
        else:
            return str(avg)

    empty_frames = df[df['age_formatted'].isna()]
    empty_frames['age_filled'] = empty_frames.apply(match_country, axis=1, args=(province_ages,country_ages,median_to_fill))
    df['age_filled'] = df['age_formatted']
    df.update(empty_frames)
    return df

def parse_dates(df):
    #fill empty dates
    date_df = df['date_confirmation'].value_counts().idxmax(skipna=True)
    filled_df = df['date_confirmation'].fillna(str(date_df))
    df.update(filled_df)
    #parse strange date formats
    def split_date(row):
        date = row['date_confirmation']
        if '-' in date:
            return date.split('-')[0]
        return row['date_confirmation']

    df['new_date_confirmation'] = df.apply(split_date, axis=1)
    return df

def fill_country_province(df):
    country_na = df[df['country'].isna()]
    geolocator = Nominatim(user_agent="Cmpt459")
    def find_country(row):
        lat = row['latitude']
        long = row['longitude']
        coords = str(lat) + ", " + str(long)
        addr = geolocator.reverse(coords, language="en")
        return addr[0].split()[-1]
    country_na['country'] = country_na.apply(find_country, axis=1)
    df['country_filled'] = df['country']
    df.update(country_na)
    province_na = df[df['province'].isna()]
    def find_province(row):
        lat = row['latitude']
        long = row['longitude']
        coords = str(lat) + ", " + str(long)
        addr = geolocator.reverse(coords, language="en")
        if addr == None:
            return np.nan

        addr_split = addr[0].split(',')
        if len(addr_split) < 2:
            return addr_split[-1]
        else:
            return addr_split[-2]

    province_na['province_filled'] = province_na.apply(find_province,axis=1)
    df['province_filled'] = df['province']
    df.update(province_na)
    return df

#drop missing lat and long rows
dropLatLong = df.dropna(subset=['latitude','longitude'])

print('Filling missing genders...')
filled_gender_df = find_gender_outcome(dropLatLong)

print('Filling missing countries and ages...')
filled_country_df = fill_country_province(filled_gender_df)

print('Filling missing ages...')
filled_age_df = process_age(filled_country_df)

print('Filling missing dates...')
clean_date_df = parse_dates(filled_age_df)
#
clean_date_df.to_csv('./cleaned_cases_train.csv',index=False)



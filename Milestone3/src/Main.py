# CMPT459 Data Mining
# Spring 2021 Milestone 3
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import RandomForests
from imblearn.over_sampling import SMOTENC
from collections import Counter

epoch = datetime.datetime.utcfromtimestamp(0)
def unix_time_millis(row):
    dt = row['new_date_confirmation'].strip()
    dt = datetime.datetime.strptime(dt, '%d.%m.%Y')
    return (dt - epoch).total_seconds()

def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('./data/cases_train_processed.csv')
    df['date_int'] = df.apply(unix_time_millis, axis=1)
    random_forest(df)

def random_forest(df):
    le = LabelEncoder()

    # Temporary grid of 3 hyperparameters for hyperparameter tuning
    param_grid = {
        "max_features": ['log2'],
        "max_depth": [50, 60, 70, 80, 90,100],
        "min_samples_split":[2],
        "min_samples_leaf": [2,3,4],
        "n_estimators": [30,40,50,60,70,80,90,100,110,120,130]
    }


    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio','date_int']]
    outcomes = df['outcome']

    all_data['filled_sex_bin'] = le.fit_transform(all_data['filled_sex'])
    all_data['province_filled_bin'] = le.fit_transform(all_data['province_filled'])
    all_data['country_filled_bin'] = le.fit_transform(all_data['country_filled'])
    new_data = all_data.drop(columns=['filled_sex', 'province_filled',
                'country_filled'])

    categories = all_data[['filled_sex_bin','province_filled_bin','country_filled_bin']]


    train_x, validate_x, train_y, validate_y = train_test_split(new_data, outcomes, test_size=0.2, random_state=42, shuffle=True)
    encoder = OneHotEncoder(categories = 'auto', handle_unknown='error', dtype=np.uint8)
    encoder.fit(categories)

    smotenc = SMOTENC([8,9,10],random_state = 101,sampling_strategy={'deceased': 99847})
    X,y = smotenc.fit_resample(train_x, train_y)
    x_dataframe = pd.DataFrame(X, columns=['age_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio','date_int','filled_sex_bin','province_filled_bin',
                'country_filled_bin'])

    y_dataframe = pd.DataFrame(y,columns=['outcome'])
    print(x_dataframe.dtypes)

    counter = Counter(y_dataframe['outcome'])
    print(counter)

    tmp = x_dataframe[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]
    stripped_dataframe = x_dataframe.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    train_x_encoded = encoder.transform(tmp).toarray()
    encoded = pd.DataFrame(train_x_encoded, index=stripped_dataframe.index)

    true_train_x = pd.concat([stripped_dataframe,encoded], axis=1)


    # 2.2 Training Model
    print("Training Random Forests")
    RandomForests.rf_train(true_train_x, y_dataframe, param_grid)

    
    # 2.3 Evaluate performance
    print("Evaluating Random Forests Training")
    tmp = train_x[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]

    stripped_dataframe = train_x.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    train_x_encoded = encoder.transform(tmp).toarray()
    encoded = pd.DataFrame(train_x_encoded, index=stripped_dataframe.index)
    true_train_x = pd.concat([stripped_dataframe,encoded], axis=1)
    RandomForests.rf_eval(true_train_x, train_y, True)

    print("Evaluating Random Forests Validation")
    validate_tmp = validate_x[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]
    validate_strip = validate_x.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    validate_x_encoded = encoder.transform(validate_tmp).toarray()
    print(validate_strip)

    new_encoded = pd.DataFrame(validate_x_encoded,index=validate_strip.index)
    true_validate_x = pd.concat([validate_strip,new_encoded], axis=1)
    RandomForests.rf_eval(true_validate_x, validate_y, False)


if __name__ == '__main__':
    main()

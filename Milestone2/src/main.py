# CMPT459 Data Mining
# Spring 2021 Milestone 2
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
# import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle

import RandomForests

from Milestone2.src.RandomForests import check_deaths


def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('../data/cases_train_processed.csv')

    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]

    # One hot encoding for categorical values
    category_clean = pd.get_dummies(all_data)

    # Attach outcome column back
    category_clean['outcome'] = df['outcome']
    category_clean['age_filled'] = df['age_filled']
    category_clean['filled_sex'] = df['filled_sex']
    category_clean['province_filled'] = df['province_filled']
    category_clean['country_filled'] = df['country_filled']

    print("Splitting data into training and validation sets")
    train, validate = train_test_split(category_clean, test_size=0.2, random_state=42, shuffle=True)

    train_attr = train.drop(columns=['outcome', 'age_filled','filled_sex','province_filled','country_filled']) # Features
    train_outcomes = train[['outcome']]

    v_data = validate.drop(columns=['outcome', 'age_filled','filled_sex','province_filled','country_filled'])
    v_outcomes = validate[['outcome']]
    '''
    print("Training Random Forests")
    RandomForests.rf_train(train_attr, train_outcomes)
    print("Evaluating Random Forests")
    RandomForests.rf_eval(v_data,v_outcomes)
    '''
    check_deaths(validate)
if __name__ == '__main__':
    main()

# CMPT459 Data Mining
# Spring 2021 Milestone 2
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

import RandomForests, LightGbm

def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('../data/cases_train_processed.csv')

    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]

    # # Random Forest Classification
    # # One hot encoding for categorical values
    # category_clean = pd.get_dummies(all_data)

    # # Attach outcome column back
    # category_clean['outcome'] = df['outcome']

    # print("Splitting data into training and validation sets")
    # train, validate = train_test_split(category_clean, test_size=0.2, random_state=42, shuffle=True)

    # v_data = validate.drop(columns=['outcome'])
    # v_outcomes = validate[['outcome']]
    # print("Training Random Forests")
    # RandomForests.rf_train(train_attr, train_outcomes)
    # print("Evaluating Random Forests")
    # RandomForests.rf_eval(v_data,v_outcomes)

    # LightGBM Classification
    y = df[['outcome']]
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(all_data, y_encoded, test_size=0.2, random_state=42, shuffle=True)

    # print("Training LightGBM")
    # LightGbm.boosted_train(X_train, y_train)
    print("Evaluating LightGBM")
    LightGbm.boosted_eval(X_valid, y_valid)

if __name__ == '__main__':
    main()

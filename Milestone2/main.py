# CMPT459 Data Mining
# Spring 2021 Milestone 2
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle

import RandomForests, LightGbm


def main():
    df = pd.read_csv('results/cases_train_processed.csv')

    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]

    # One hot encoding for categorical values
    category_clean = pd.get_dummies(all_data)

    # Attach outcome column back
    category_clean['outcome'] = df['outcome']

    print("Splitting data into training and validation sets")
    train, validate = train_test_split(category_clean, test_size=0.2, random_state=42, shuffle=True)

    train_attr = train.drop(columns=['outcome']) # Features
    train_outcomes = train[['outcome']]

    print("Training Random Forests")
    RandomForests.rf_train(train_attr, train_outcomes)
    print("Evaluating Random Forests")
    RandomForests.rf_eval(validate)

if __name__ == '__main__':
    main()
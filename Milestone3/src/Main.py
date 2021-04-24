# CMPT459 Data Mining
# Spring 2021 Milestone 3
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import RandomForests, LightGbm
from imblearn.over_sampling import SMOTENC
from collections import Counter

def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('../data/cases_train_processed.csv')

    # random_forest(df)
    light_gbm(df)

def random_forest(df):

    # Temporary grid of 3 hyperparameters for hyperparameter tuning
    param_grid = {
        "max_depth": [50, 60, 70, 80, 90],
        "min_samples_leaf": [2, 3, 4],
        "n_estimators": [100, 300, 500, 600]
    }


    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]
    outcomes = df['outcome']
    counter = Counter(outcomes)
    print(counter)

    #Smotenc part
    # the [0,1,2,3] are the index of which columns hold categorical values if im not wrong
    smotenc = SMOTENC([0,1,2,3],random_state = 101, sampling_strategy={'deceased': 99847})
    X,y = smotenc.fit_resample(all_data, outcomes)
    x_dataframe = pd.DataFrame(X, columns=['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio'])
    y_dataframe = pd.DataFrame(y,columns=['outcome'])



    train_x, validate_x, train_y, validate_y = train_test_split(x_dataframe, y_dataframe, test_size=0.2, random_state=42, shuffle=True)
    encoder = OneHotEncoder(categories = "auto")
    encoder.fit(x_dataframe)

    train_x_encoded = encoder.transform(train_x)
    

    # 2.2 Training Model
    print("Training Random Forests")
    RandomForests.rf_train(train_x_encoded, train_y, param_grid)


    # 2.3 Evaluate performance
    print("Evaluating Random Forests Training")
    RandomForests.rf_eval(train_x_encoded, train_y, True)

    print("Evaluating Random Forests Validation")
    validate_x_encoded = encoder.transform(validate_x)
    RandomForests.rf_eval(validate_x_encoded,validate_y,False)


def light_gbm(df):
    # Temporary grid of 3 hyperparameters for hyperparameter tuning
    param_grid = {
        "boosting_type": ['gbdt'], #GradientBoostingDecisionTree
        "objective": ['multiclass'],
        "metric": ['multi_logloss'],
        "num_class": [4],
        # "learning_rate": [0.03, 0.05],
        # "max_depth": [5, 10],
        # "num_leaves": [20, 30, 40],
        # "n_estimators": [100, 300]
        "learning_rate": [0.03],
        "max_depth": [5],
        "num_leaves": [20, 30],
        "n_estimators": [100]
    }
    X = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]
    y = df[['outcome']]
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)
    #Smotenc part
    # the [0,1,2,3] are the index of which columns hold categorical values if im not wrong
    deceased_encoded = le.transform(['deceased'])[0]

    # smotenc = SMOTENC([1,2,3],random_state = 101, sampling_strategy={0: 99847})
    smotenc = SMOTENC([1,2,3],random_state = 101, sampling_strategy={0: 5000})
    X_train,y_train = smotenc.fit_resample(X_train, y_train)


    # 2.2 Train Model
    LightGbm.lightgbm_train(X_train, y_train, param_grid, le)
    LightGbm.lightgbm_check_model_stats()
    # 2.3 Evaluate performance
    LightGbm.lightgbm_eval(X_train, y_train, le, "train")
    LightGbm.lightgbm_eval(X_valid, y_valid, le, "valid")

if __name__ == '__main__':
    main()

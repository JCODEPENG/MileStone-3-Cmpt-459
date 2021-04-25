# CMPT459 Data Mining
# Spring 2021 Milestone 3
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import RandomForests, LightGbm
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
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

    df = pd.read_csv('../data/cases_train_processed.csv')
    df['date_int'] = df.apply(unix_time_millis, axis=1)
    print(df)

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
        "learning_rate": [0.03, 0.05, 0.07],
        "max_depth": [10, 15, 20],
        "num_leaves": [30, 40, 50],
        "n_estimators": [200, 300, 400]
        # "learning_rate": [0.05],
        # "max_depth": [10],
        # "num_leaves": [40],
        # "n_estimators": [300]
    }

    X_train, X_valid, y_train, y_valid, le = get_oversampled_encoded_data(df)

    print("\n--------------TRAINING MODEL--------------\n")
    LightGbm.lightgbm_train(X_train, y_train, param_grid, le)
    print("\n--------------CHECKING MODEL STATS--------------\n")
    LightGbm.lightgbm_check_model_stats()
    print("\n--------------EVALUATING MODEL ON TRAINING DATA--------------\n")
    LightGbm.lightgbm_eval(X_train, y_train, le, "train")
    print("\n--------------EVALUATING MODEL ON VALIDATION DATA--------------\n")
    LightGbm.lightgbm_eval(X_valid, y_valid, le, "valid")

def get_oversampled_encoded_data(df):
    split_data = False
    X = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio', 'date_int']]
    y = df[['outcome']]
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    if (split_data):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)
    else:
        X_train = X
        y_train = y_encoded
        X_valid = X
        y_valid = y_encoded

    #Smotenc part
    # the [0,1,2,3] are the index of which columns hold categorical values if im not wrong
    deceased_encoded = le.transform(['deceased'])[0]

    ## over/undersampling code
    # hospitalized_encoded = le.transform(['hospitalized'])[0]
    # nonhospitalized_encoded = le.transform(['nonhospitalized'])[0]
    # recovered_encoded = le.transform(['recovered'])[0]

    # smotenc = SMOTENC([1,2,3],random_state = 101, sampling_strategy={0: 99847})
    # over = SMOTENC([1,2,3],random_state = 101, sampling_strategy={deceased_encoded: 37500})
    # under = RandomUnderSampler(sampling_strategy={hospitalized_encoded: 37500, nonhospitalized_encoded: 37500, recovered_encoded: 37500})
    # steps = [('o', over), ('u', under)]
    # pipeline = Pipeline(steps=steps)
    # X_train, y_train = pipeline.fit_resample(X_train, y_train)

    smotenc = SMOTENC([1,2,3],random_state = 101, sampling_strategy={deceased_encoded: 37500})
    X_train,y_train = smotenc.fit_resample(X_train, y_train)

    return X_train, X_valid, y_train, y_valid, le

if __name__ == '__main__':
    main()

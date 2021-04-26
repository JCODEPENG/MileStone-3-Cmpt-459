# CMPT459 Data Mining
# Spring 2021 Milestone 3
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
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

    # load datasets
    df = pd.read_csv('../data/cases_train_processed.csv')
    df['date_int'] = df.apply(unix_time_millis, axis=1)
    test_df = pd.read_csv('../data/cases_test_processed.csv')
    test_df['date_int'] = test_df.apply(unix_time_millis, axis=1)

    # LightGBM Training
    split_data = False
    if (split_data):
        print("splitting training dataset")
        X_train, X_valid, y_train, y_valid, le = get_oversampled_encoded_data(df, split_data) # 80/20 split
    else:
        print("using entire training dataset")
        X_train, _,        y_train, _,       le = get_oversampled_encoded_data(df, split_data) # full dataset
    
    light_gbm_train(X_train, y_train, le)

    # LightGBM Generate Testset Predictions
    X_test = get_relevant_columns(test_df)
    light_gbm_eval(X_test, _, le, "test")

    # Random Forest
    # random_forest(df)


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
    print("\n--------------TRAINING MODEL--------------\n")
    RandomForests.rf_train(true_train_x, y_dataframe, param_grid)

    
    # 2.3 Evaluate performance
    print("\n--------------EVALUATING MODEL ON TRAINING DATA--------------\n")
    tmp = train_x[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]

    stripped_dataframe = train_x.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    train_x_encoded = encoder.transform(tmp).toarray()
    encoded = pd.DataFrame(train_x_encoded, index=stripped_dataframe.index)
    true_train_x = pd.concat([stripped_dataframe,encoded], axis=1)
    RandomForests.rf_eval(true_train_x, train_y, True)

    print("\n--------------EVALUATING MODEL ON VALIDATION DATA--------------\n")

    validate_tmp = validate_x[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]
    validate_strip = validate_x.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    validate_x_encoded = encoder.transform(validate_tmp).toarray()
    print(validate_strip)

    new_encoded = pd.DataFrame(validate_x_encoded,index=validate_strip.index)
    true_validate_x = pd.concat([validate_strip,new_encoded], axis=1)
    RandomForests.rf_eval(true_validate_x, validate_y, False)



def light_gbm_train(X_train, y_train, le):
    # Temporary grid of 3 hyperparameters for hyperparameter tuning
    param_grid = {
        "boosting_type": ['gbdt'], #GradientBoostingDecisionTree
        "objective": ['multiclass'],
        "metric": ['multi_logloss'],
        "num_class": [4],
        # "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.9, 0.11],
        # "max_depth": [6, 8, 10, 12],
        # "num_leaves": [25, 30, 35, 40, 45],
        "learning_rate": [0.11],
        "max_depth": [12],
        "num_leaves": [45],
    }
    print("\n--------------TRAINING MODEL--------------\n")
    LightGbm.lightgbm_train(X_train, y_train, param_grid, le)
    print("\n--------------CHECKING MODEL STATS--------------\n")
    LightGbm.lightgbm_check_model_stats()

def light_gbm_eval(X, y, le, dataset):
    if (dataset == "train"):
        print("\n--------------EVALUATING MODEL ON TRAINING DATA--------------\n")
        LightGbm.lightgbm_eval(X, y, le, "train")
    elif (dataset == "valid"):
        print("\n--------------EVALUATING MODEL ON VALIDATION DATA--------------\n")
        LightGbm.lightgbm_eval(X, y, le, "valid")
    elif (dataset == "test"):
        print("\n--------------EVALUATING MODEL ON TEST DATA--------------\n")
        LightGbm.lightgbm_eval(X, y, le, "test")


def get_relevant_columns(df):
    X = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio', 'date_int']]
    return X

def get_oversampled_encoded_data(df, split_data):
    X = get_relevant_columns(df)
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

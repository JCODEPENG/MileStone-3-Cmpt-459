# CMPT459 Data Mining
# Spring 2021 Milestone 3
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import RandomForests
from imblearn.over_sampling import SMOTENC
from collections import Counter

def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('./data/cases_train_processed.csv')

    random_forest(df)

def random_forest(df):
    le = LabelEncoder()

    # Temporary grid of 3 hyperparameters for hyperparameter tuning
    param_grid = {
        "max_depth": [50, 60, 70, 80, 90],
        "min_samples_leaf": [2, 3, 4],
        "n_estimators": [100, 300, 500, 600]
    }


    all_data = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]
    print(pd.get_dummies(all_data))
    outcomes = df['outcome']

    all_data['filled_sex_bin'] = le.fit_transform(all_data['filled_sex'])
    all_data['province_filled_bin'] = le.fit_transform(all_data['province_filled'])
    all_data['country_filled_bin'] = le.fit_transform(all_data['country_filled'])
    print(all_data['filled_sex_bin'])
    new_data = all_data.drop(columns=['filled_sex', 'province_filled',
                'country_filled'])

    categories = all_data[['filled_sex_bin','province_filled_bin','country_filled_bin']]



    train_x, validate_x, train_y, validate_y = train_test_split(new_data, outcomes, test_size=0.2, random_state=42, shuffle=True)
    encoder = OneHotEncoder(categories = 'auto', handle_unknown='error', dtype=np.uint8)
    encoder.fit(categories)
    counter = Counter(train_y)
    print(counter)


    # the [1,2,3] are the index of which columns hold categorical values if im not wrong
    smotenc = SMOTENC([7,8,9],random_state = 101,sampling_strategy={'deceased': 99847})
    X,y = smotenc.fit_resample(train_x, train_y)
    x_dataframe = pd.DataFrame(X, columns=['age_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio','filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    print(x_dataframe)
    y_dataframe = pd.DataFrame(y,columns=['outcome'])


    tmp = x_dataframe[['filled_sex_bin','province_filled_bin',
                'country_filled_bin']]
    print(tmp)
    stripped_dataframe = x_dataframe.drop(columns=['filled_sex_bin','province_filled_bin',
                'country_filled_bin'])
    train_x_encoded = encoder.transform(tmp).toarray()
    encoded = pd.DataFrame(train_x_encoded, index=stripped_dataframe.index)
    print(encoded)

    true_train_x = pd.concat([stripped_dataframe,encoded], axis=1)

    print(true_train_x)
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
    print(encoded)
    true_train_x = pd.concat([stripped_dataframe,encoded], axis=1)
    print(true_train_x)
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
    RandomForests.rf_eval(true_validate_x,validate_y,False)

    """
    RandomForests.investigate_deaths(validate)

    # 2.4 Vary hyperparameter and check for overfitting
    train_scores = []
    validation_scores = []
    depth_values = range(10,110,10)
    for depth in depth_values:
         print("Training RandomForest with depth", depth)
         clf = RandomForests.overfit_rf_train(train_attr, train_outcomes, depth)
         train_accuracy = RandomForests.overfit_eval(train_attr, train_outcomes, clf)
         train_scores.append(train_accuracy)

         validation_accuracy = RandomForests.overfit_eval(v_data, v_outcomes, clf)
         validation_scores.append(validation_accuracy)
    plt.figure()
    plt.plot(depth_values, train_scores)
    plt.plot(depth_values, validation_scores)
    plt.title("Accuracy vs Max Depth Hyperparameter for Random Forests")
    plt.ylabel("Accuracy")
    plt.xlabel("Max Depth Hyperparameter")
    plt.legend(['training scores', 'validation scores'])
    plt.savefig("../plots/overfitting_check_rf.png")



def light_gbm(df):
    X = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]
    y = df[['outcome']]
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

    # 2.2 Train Model
    LightGbm.boosted_train(X_train, y_train, 8)

    # 2.3 Evaluate performance
    LightGbm.boosted_eval(X_train, y_train, le, True, True)
    LightGbm.boosted_eval(X_valid, y_valid, le, True, False)

    # Find feature importance
    LightGbm.boosted_feature_importance(X_train)

    # 2.4 Vary hyperparameter and check for overfitting
    train_scores = []
    validation_scores = []
    depth_values = range(2,20,2)
    for depth in depth_values:
        print("Training LightGBM with depth", depth)
        LightGbm.boosted_train(X_train, y_train, depth)
        train_accuracy = LightGbm.boosted_eval(X_train, y_train, le, False)
        train_scores.append(train_accuracy)

        validation_accuracy = LightGbm.boosted_eval(X_valid, y_valid, le, False)
        validation_scores.append(validation_accuracy)
    plt.figure()
    plt.plot(depth_values, train_scores)
    plt.plot(depth_values, validation_scores)
    plt.title("Accuracy vs Max Depth Hyperparameter for LightGBD")
    plt.ylabel("Accuracy")
    plt.xlabel("Max Depth Hyperparameter")
    plt.legend(['training scores', 'validation scores'])
    plt.savefig("../plots/overfitting_check_gbd.png", bbox_inches = "tight")

"""

if __name__ == '__main__':
    main()

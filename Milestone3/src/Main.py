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

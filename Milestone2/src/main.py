# CMPT459 Data Mining
# Spring 2021 Milestone 2
# Joshua Peng & Lucia Schmidt

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

import RandomForests, LightGbm


def main():
    directory = os.path.dirname('../models/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.read_csv('../data/cases_train_processed.csv')

    # random_forest(df)
    light_gbm(df)

def random_forest(df):
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

    print("Training Random Forests")
    RandomForests.rf_train(train_attr, train_outcomes)
    print("Evaluating Random Forests")
    RandomForests.rf_eval(v_data,v_outcomes)

    RandomForests.investigate_deaths(validate)


def light_gbm(df):
    X = df[['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']]
    y = df[['outcome']]
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

    # 2.3 Evaluate performance
    LightGbm.boosted_train(X_train, y_train, 8)
    train_accuracy = LightGbm.boosted_eval(X_train, y_train, le, True)
    validation_accuracy = LightGbm.boosted_eval(X_valid, y_valid, le, True)

    # Find feature importance
    LightGbm.boosted_feature_importance(X_train)

    # # 2.4 Vary hyperparameter and check for overfitting
    # train_scores = []
    # validation_scores = []
    # depth_values = range(2,20,2)
    # for depth in depth_values:
    #     print("Training LightGBM with depth", depth)
    #     LightGbm.boosted_train(X_train, y_train, depth)
    #     train_accuracy = LightGbm.boosted_eval(X_train, y_train, le, False)
    #     train_scores.append(train_accuracy)

    #     validation_accuracy = LightGbm.boosted_eval(X_valid, y_valid, le, False)
    #     validation_scores.append(validation_accuracy)
    # plt.figure()
    # plt.plot(depth_values, train_scores)
    # plt.plot(depth_values, validation_scores)
    # plt.title("Accuracy vs Max Depth Hyperparameter for LightGBD")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Max Depth Hyperparameter")
    # plt.legend(['training scores', 'validation scores'])
    # plt.savefig("../plots/overfitting_check_gbd.png")


if __name__ == '__main__':
    main()

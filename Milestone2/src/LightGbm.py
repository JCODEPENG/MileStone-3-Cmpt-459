from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle

filename = '../models/boosted_classifier.pkl'
feature_name = ['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']
categorical_feature = ['filled_sex', 'province_filled', 'country_filled']


def boosted_train(data, label):
    # Set the params
    params={}
    params['learning_rate']=0.03
    params['boosting_type']='gbdt' #GradientBoostingDecisionTree
    params['objective']='multiclass' #Binary target feature
    params['metric']='multi_logloss' 
    params['max_depth']=10
    params['num_class'] = 4
    
    epochs=100

    for col in categorical_feature:
        data[col] = data[col].astype('category')

    # Train model and write to file
    print("Converting dataset to lgb format")
    train_data = lgb.Dataset(data, label=label, feature_name=feature_name, categorical_feature=categorical_feature)
    print("Training LightGBM classifier")
    clf = lgb.train(params, train_data, epochs)
    print("Saving LightGBM model")
    pickle.dump(clf, open(filename, 'wb'))

def boosted_eval(X, y):
    for col in categorical_feature:
        X[col] = X[col].astype('category')
    # Load Model
    clf_load = pickle.load(open(filename, 'rb'))
    predictions = clf_load.predict(X)
    predictions = [np.argmax(line) for line in predictions]
    value = accuracy_score(y, predictions)


    print("ACCURACY: ", value)

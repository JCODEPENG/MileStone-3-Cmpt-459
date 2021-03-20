from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

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

def boosted_eval(X, y, le):
    for col in categorical_feature:
        X[col] = X[col].astype('category')
    # Load Model
    clf_load = pickle.load(open(filename, 'rb'))
    predictions = clf_load.predict(X)
    predictions = [np.argmax(line) for line in predictions]
    value = accuracy_score(y, predictions)
    print("ACCURACY: ", value)

    outcomes = le.inverse_transform(y)
    predictions = le.inverse_transform(predictions)

    # Confusion Matrix
    # Modified from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
    matrix = confusion_matrix(outcomes, predictions)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2)

    class_names = np.unique(outcomes)
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for LightGBM Model')
    plt.show()

    # F1-scores
    print(classification_report(outcomes, predictions))

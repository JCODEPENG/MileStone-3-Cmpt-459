from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import tqdm

filename = 'rf_classifier.pkl'
estimators = 100
def rf_train(train_attr, train_outcomes):
    clf = RandomForestClassifier(n_estimators=estimators)

    # Train model and write to file
    clf.fit(train_attr, train_outcomes)
    pickle.dump(clf, open(filename, 'wb'))

def rf_eval(validate):
    # Load Model
    v_data = validate.drop(columns=['outcome'])
    v_outcomes = validate[['outcome']]
    clf_load = pickle.load(open(filename, 'rb'))
    predictions = clf_load.predict(v_data)
    value = accuracy_score(v_data, predictions)

    print("ACCURACY: ", end=" ")
    print(value)


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
# import tqdm

filename = '../models/rf_classifier.pkl'
estimators = 50
def rf_train(train_attr, train_outcomes):
    clf = RandomForestClassifier(n_estimators=estimators)

    # Train model and write to file
    clf.fit(train_attr, train_outcomes)
    pickle.dump(clf, open(filename, 'wb'))

def rf_eval(data, outcomes):
    clf_load = pickle.load(open(filename, 'rb'))
    predictions = clf_load.predict(data)
    value = accuracy_score(outcomes, predictions)
    print("Accuracy Score: ", end=" ")
    print(value)

    # Confusion Matrix
    # Modified from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
    matrix = confusion_matrix(outcomes, predictions)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2)

    class_names = np.unique(outcomes['outcome'])
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()
    #

    # F1-scores
    print(classification_report(outcomes, predictions))

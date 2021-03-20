from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

def buildModel(train_attr, train_outcomes, filename):
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(train_attr, train_outcomes)
    pickle.dump(clf, open(filename, 'wb'))

df = pd.read_csv('./results/cases_train_processed.csv')

all_data = df[['age_filled', 'filled_sex', 'province_filled',
             'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
             'Incidence_Rate', 'Case-Fatality_Ratio']]

# One hot encoding for categorical values
category_clean = pd.get_dummies(all_data)

# Attach outcome column back
category_clean['outcome'] = df['outcome']

train, validate = train_test_split(category_clean, test_size=0.2, random_state=43, shuffle=True)

train_attr = train.drop(columns=['outcome']) # Features
train_outcomes = train[['outcome']]

filename = 'rf_classifier.pkl'

v_data = validate.drop(columns=['outcome'])
v_outcomes = validate[['outcome']]


def evaluateModel(filename, data, outcomes):
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


#print("Build Random Forest Model")
#buildModel(train_attr, train_outcomes, filename)

print("Evaluating on training data")
evaluateModel(filename, train_attr, train_outcomes)

print("Evaluating on validation data")
evaluateModel(filename,v_data,v_outcomes)

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('./results/cases_train_processed.csv')

all_data = df[['age_filled', 'filled_sex', 'province_filled',
             'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
             'Incidence_Rate', 'Case-Fatality_Ratio']]

clf = RandomForestClassifier(n_estimators=1000)

# One hot encoding for categorical values
category_clean = pd.get_dummies(all_data)

# Attach outcome column back
category_clean['outcome'] = df['outcome']

train, validate = train_test_split(category_clean, test_size=0.2, random_state=42, shuffle=True)

train_attr = train.drop(columns=['outcome']) # Features
train_outcomes = train[['outcome']]

filename = 'rf_classifier.pkl'

# Train model and write to file
clf.fit(train_attr, train_outcomes)
pickle.dump(clf, open(filename, 'wb'))

v_data = validate.drop(columns=['outcome'])
v_outcomes = validate[['outcome']]

# Load Model
clf_load = pickle.load(open(filename, 'rb'))
predictions = clf_load.predict(v_data)
value = accuracy_score(v_data, predictions)

print("ACCURACY: ", end=" ")
print(value)


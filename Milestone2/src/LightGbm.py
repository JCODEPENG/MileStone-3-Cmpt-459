from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle



# # Train model and write to file
# clf.fit(train_attr, train_outcomes)
# pickle.dump(clf, open(filename, 'wb'))

# v_data = validate.drop(columns=['outcome'])
# v_outcomes = validate[['outcome']]

# # Load Model
# clf_load = pickle.load(open(filename, 'rb'))
# predictions = clf_load.predict(v_data)
# value = accuracy_score(v_data, predictions)

# print("ACCURACY: ", end=" ")
# print(value)


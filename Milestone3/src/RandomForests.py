import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

filename = '../models/rf_classifier.pkl'

def f1Deceased(val, predict):
    ans = precision_recall_fscore_support(val,predict, labels=['deceased'])
    return ans[2]

def recallDeceased(val, predict):
    ans = precision_recall_fscore_support(val,predict, labels=['deceased'])
    return ans[1]

def rf_train(train_attr, train_outcomes, param_grid):

    scoring = {
            'f1_deceased': make_scorer(f1Deceased),
            'recall_deceased': make_scorer(recallDeceased),
            'accuracy': make_scorer(accuracy_score),
            'recall': make_scorer(recall_score, average = 'micro')}


    clf = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(clf, param_grid, n_iter=32,
                                        scoring=scoring, refit='accuracy', verbose=10, cv=5,
                                        n_jobs=1, random_state=42)


    # Train model and write to file
    random_search.fit(train_attr, train_outcomes)
    all_results = random_search.cv_results_

    for i in range(0,len(all_results['params'])):
        print("Combination: ", end="")
        print(all_results['params'][i])
        print("F1 deceased score: ", end="")
        print(all_results['mean_test_f1_deceased'][i])
        print("Recall deceased score: ", end="")
        print(all_results['mean_test_recall_deceased'][i])
        print("Overall accuracy score: ", end="")
        print(all_results['mean_test_accuracy'][i])
        print("Overall recall score: ", end="")
        print(all_results['mean_test_recall'][i])
        print()


    print(random_search.best_params_)
    print(random_search.best_score_)
    # clf.fit(train_attr, train_outcomes)
    # pickle.dump(clf, open(filename, 'wb'))


def overfit_rf_train(train_attr, train_outcomes, estimators):
    clf = RandomForestClassifier(n_estimators=estimators)
    # Train model and write to file
    clf.fit(train_attr, train_outcomes)
    return clf

def overfit_eval(data,outcomes, clf):
    predictions = clf.predict(data)
    value = accuracy_score(outcomes, predictions)
    return value

def rf_eval(data, outcomes, name):
    clf_load = pickle.load(open(filename, 'rb'))
    predictions = clf_load.predict(data)
    value = accuracy_score(outcomes, predictions)
    print("Accuracy Score: ", end=" ")
    print(value)

'''
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
    if(name):
        plt.title('Confusion Matrix for Random Forest Model - Train Data')
        plt.savefig("../plots/confusion_matrix_train_rf.png", bbox_inches = "tight")
    else:
        plt.title('Confusion Matrix for Random Forest Model - Validation Data')
        plt.savefig("../plots/confusion_matrix_val_rf.png", bbox_inches = "tight")
    #

    # F1-scores
    print(classification_report(outcomes, predictions))
'''

def investigate_deaths(validate):
    clf_load = pickle.load(open(filename, 'rb'))

    feature_list = ['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']

     # feature importance adapted from https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm
    feature_imp = pd.DataFrame(sorted(zip(clf_load.feature_importances_, feature_list)),columns=['Value','Feature'])
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('Random Forest Features Importance')
    plt.tight_layout()
    plt.savefig('../plots/feature_importances_rf.png')

    deaths = validate[validate['outcome'] == 'deceased']
    hospitalized = validate[validate['outcome'] == 'hospitalized']
    hospitalized_avgs = {'age': hospitalized['age_filled'].mean(), 'gender': hospitalized['filled_sex'].mode()[0],
                         'province_filled': hospitalized['province_filled'].mode()[0],
                         'country_filled': hospitalized['country_filled'].mode()[0], 'Confirmed': hospitalized['Confirmed'].mode()[0],
                         'Deaths': hospitalized['Deaths'].mode()[0], 'Recovered': hospitalized['Recovered'].mode()[0],
                         'Incidence_Rate': hospitalized['Incidence_Rate'].mode()[0],
                         'Active': hospitalized['Active'].mode()[0]}

    overallCount = 0
    for idx, row in deaths.iterrows():
        matches = 0
        if row['province_filled'] == hospitalized_avgs['province_filled']:
            matches+=1
        if row['Recovered'] >= hospitalized_avgs['Recovered'] - 10000 and row['Recovered'] <= hospitalized_avgs['Recovered'] + 10000:
            matches +=1
        if row['Deaths'] >= hospitalized_avgs['Deaths'] - 1000 and row['Deaths'] <= hospitalized_avgs['Deaths'] + 1000:
            matches +=1
        if row['Active'] >= hospitalized_avgs['Active'] - 10000 and row['Active'] <= hospitalized_avgs['Active'] + 10000:
            matches +=1
        if matches > 3:
            overallCount+=1

    print('Similarity between deceased and hospitalized: ', end=" ")
    print(round(overallCount/len(deaths['outcome']), 2))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


filename = '../models/lightgbm_classifier.pkl'
feature_name = ['age_filled', 'filled_sex', 'province_filled',
                'country_filled','Confirmed', 'Deaths', 'Recovered','Active',
                'Incidence_Rate', 'Case-Fatality_Ratio']
categorical_feature = ['filled_sex', 'province_filled', 'country_filled']

def f1Deceased(val, predict):
    ans = precision_recall_fscore_support(val,predict, labels=[0])
    return ans[2]

def recallDeceased(val, predict):
    ans = precision_recall_fscore_support(val,predict, labels=[0])
    return ans[1]

def lightgbm_train(data, label, param_grid, le):
    model = lgb.LGBMClassifier()
    deceased_encoded = le.transform(['deceased'])
    assert(deceased_encoded == [0])

    scoring = {
        'f1_deceased': make_scorer(f1Deceased),
        'recall_deceased': make_scorer(recallDeceased),
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score, average = 'micro')
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=scoring,
        refit='f1_deceased',
        verbose=10,
    )

    for col in categorical_feature:
        data[col] = data[col].astype('category')

    fitted_model = grid_search.fit(data, label)
    print("Saving LightGBM model")
    pickle.dump(fitted_model, open(filename, 'wb'))

def lightgbm_check_model_stats():
    model = pickle.load(open(filename, 'rb'))
    all_results = model.cv_results_
    results_df = []
    for i in range(0,len(all_results['params'])):
        combination = all_results['params'][i]
        f1_deceased = all_results['mean_test_f1_deceased'][i]
        recall_deceased = all_results['mean_test_recall_deceased'][i]
        overall_accuracy = all_results['mean_test_accuracy'][i]
        overall_recall = all_results['mean_test_recall'][i]
        results_df.append([combination, f1_deceased, recall_deceased, overall_accuracy, overall_recall])
        # print("Combination:", all_results['params'][i])
        # print("F1 deceased score:", all_results['mean_test_f1_deceased'][i])
        # print("Recall deceased score:", all_results['mean_test_recall_deceased'][i])
        # print("Overall accuracy score:", all_results['mean_test_accuracy'][i])
        # print("Overall recall score:", all_results['mean_test_recall'][i], "\n")
    results_df = pd.DataFrame(results_df, columns=['combination', 'f1_deceased', 'recall_deceased', 'overall_accuracy', 'overal_recall'])
    print("best params", model.best_params_)
    results_df.to_csv("lightgbm_gridsearch_results.csv")

def lightgbm_eval(X, y, le, dataset):
    for col in categorical_feature:
        X[col] = X[col].astype('category')
    # Load Model
    model = pickle.load(open(filename, 'rb')).best_estimator_

    predictions = model.predict(X)
    # predictions = [np.argmax(line) for line in predictions]
    # accuracy = accuracy_score(y, predictions)
    # print("ACCURACY: ", accuracy)
    evaluation_stats(y, predictions, le, dataset)
    

def evaluation_stats(actual, predictions, le, dataset):
    actual = le.inverse_transform(actual)
    predictions = le.inverse_transform(predictions)

    # Confusion Matrix
    # Modified from https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
    matrix = confusion_matrix(actual, predictions)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Blues, linewidths=0.2)

    class_names = np.unique(actual)
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.show()
    if (dataset == "train"):
        plt.title('Confusion Matrix for LightGBM Model - Train Data')
        plt.savefig("../plots/confusion_matrix_train_gbd.png", bbox_inches = "tight")
    elif (dataset == "valid"):
        plt.title('Confusion Matrix for LightGBM Model - Validation Data')
        plt.savefig("../plots/confusion_matrix_val_gbd.png", bbox_inches = "tight")

    # F1-scores
    print(classification_report(actual, predictions))

def boosted_feature_importance(X):
    clf = pickle.load(open(filename, 'rb'))
    # feature importance adapted from https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm 
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),X.columns)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features Importance')
    plt.tight_layout()
    plt.savefig('../plots/feature_importances_gbd.png')



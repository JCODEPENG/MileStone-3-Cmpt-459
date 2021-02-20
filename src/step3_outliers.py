import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def remove_outliers():
    df = pd.read_csv('../results/cleaned_cases_train.csv')
    plt.hist(df['age_filled'], bins=20, rwidth=0.5)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('General Age Groups')
    plt.show()

    plt.hist(df['age_filled'], bins=20, rwidth=0.5, density=True)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Normal distribution of Age Groups')
    curve = np.arange(df['age_filled'].min(),df['age_filled'].max(),0.1)
    plt.plot(curve, norm.pdf(curve,df['age_filled'].mean(),df['age_filled'].std()))
    plt.show()

    df['zscore'] = (df['age_filled']-df['age_filled'].mean())/df['age_filled'].std()
    df_age_outliers = df[(df['zscore'] > -4) & (df['zscore'] < 4)]
    clean_age_outliers = df_age_outliers.drop(columns=['zscore'])

    # Fill try to plot later
    # plt.figure()
    # plt.title('Total Provinces')
    # plt.ylabel('Count')
    # plt.xlabel('Provinces')
    # plt['provinces_filled'].hist()

    clean_province_outliers = clean_age_outliers.dropna(subset=['province_filled'])
    clean_province_outliers.to_csv("../results/cleaned_outliers_train.csv", index=False)




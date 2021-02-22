import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def remove_outliers():
    # normal distribution method referenced from https://www.youtube.com/watch?v=KFuEAGR3HS4&ab_channel=codebasics for age
    df = pd.read_csv('../results/cleaned_cases_train.csv')
    plt.hist(df['age_filled'], bins=20, rwidth=0.5)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('General Age Groups')
    plt.savefig('../plots/age_plot.png')
    plt.cla()

    plt.hist(df['age_filled'], bins=20, rwidth=0.5, density=True)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Normal distribution of Age Groups')
    curve = np.arange(df['age_filled'].min(),df['age_filled'].max(),0.1)
    plt.plot(curve, norm.pdf(curve,df['age_filled'].mean(),df['age_filled'].std()))
    plt.savefig('../plots/age_normal.png')
    plt.cla()

    df['zscore'] = (df['age_filled']-df['age_filled'].mean())/df['age_filled'].std()

    df_age_outliers = df[(df['zscore'] > -4) & (df['zscore'] < 4)]
    clean_age_outliers = df_age_outliers.drop(columns=['zscore'])
    #
    print ("The outlier columns for age: ")
    print (df[(df['zscore'] < -4) | (df['zscore'] > 4)])

    print("The outlier count for provinces (nan value count for province)")
    print(str(clean_age_outliers['province_filled'].isna().sum()) + " outliers for province_filled")

    clean_province_outliers = clean_age_outliers.dropna(subset=['province_filled'])
    clean_province_outliers.to_csv("../results/cleaned_outliers_train.csv", index=False)




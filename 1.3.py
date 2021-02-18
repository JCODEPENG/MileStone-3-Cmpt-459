import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv('./cleaned_cases_train.csv')
plt.hist(df['age'], bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.hist(df['age'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Age')
plt.ylabel('Count')
curve = np.arange(df['age'].min(),df['age'].max(),0.1)
plt.plot(curve, norm.pdf(curve,df['age'].mean(),df['age'].std()))
plt.show()

df['zscore'] = (df['age']-df['age'].mean())/df['age'].std()
df_age_outliers = df[df['zscore'] > -3]
df_age_outliers = df_age_outliers[df[df_age_outliers['zscore']] < 4]

print(df_age_outliers)

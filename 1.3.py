import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv('./cleaned_cases_train.csv')
plt.hist(df['age'], bins=20, rwidth=0.5)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('General Age Groups')
plt.show()

plt.hist(df['age'], bins=20, rwidth=0.5, density=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Normal distribution of Age Groups')
curve = np.arange(df['age'].min(),df['age'].max(),0.1)
plt.plot(curve, norm.pdf(curve,df['age'].mean(),df['age'].std()))
plt.show()

df['zscore'] = (df['age']-df['age'].mean())/df['age'].std()
df_age_outliers = df[(df['zscore'] > -4) & (df['zscore'] < 4)]
print(df[(df['zscore'] < -4)])
clean_df = df_age_outliers.drop(columns=['zscore'])
print(clean_df)

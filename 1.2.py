import numpy as np
import pandas as pd
import random
import math

df = pd.read_csv('./cases_train.csv')

def process_age_and_gender(df):
    print (type(df['age'][0]))
    df_clean = df[df['age'].notnull()]
    ages = []
    for i in df_clean['age']:
        ages.append(i)
    idx = 0
    for i in range(0,len(df['age'])):
        if '-' in ages[idx]:
            age_range = ages[idx].split('-')
            if df['age'][i] == ages[idx]:
                idx += 1
            if age_range[1] != '':
                df.loc[i,'age'] = random.randint(int(age_range[0]), int(age_range[1]))
            else:
                df.loc[i,'age'] = int(age_range[0])
        elif '+' in ages[idx]:
            over_80 = int(ages[idx][:-1])

            if df['age'][i] == ages[idx]:
                idx += 1
            df.loc[i,'age'] = over_80
        else:
            age = int(float(ages[idx]))
            if df['age'][i] == ages[idx]:
                idx += 1
            df.loc[i,'age'] = age
        
    print(df)




    print(df_clean)
    age_gender = {}
process_age_and_gender(df)

import numpy as np
import pandas as pd
import random
import math

df = pd.read_csv('./cases_train.csv')

def process_age_and_gender(df):
    df_clean = df[df['age'].notnull()]
    for i, row in df_clean.iterrows():
        age = row['age']
        if '-' in age:
            age_range = age.split('-')
            if age_range[1] != '' or age_range[1].isnumeric():
                df_clean.at[i,'age'] = random.randint(int(age_range[0]), int(age_range[1]))
            else:
                df_clean.at[i,'age'] = int(age_range[0])
        elif '+' in age:
            df.at[i,'age'] = int(age[:-1])
        elif ' ' in age:
            age_range = age.split(' ')
            if age_range[1] != '' and age_range[1].isnumeric():
                df_clean.at[i,'age'] = random.randint(int(age_range[0]), int(age_range[1]))
            else:
                df_clean.at[i,'age'] = int(age_range[0])
        else:
            df.at[i,'age'] = int(float(age))

    df.update(df_clean)
    df.bfill(inplace=True)

process_age_and_gender(df)

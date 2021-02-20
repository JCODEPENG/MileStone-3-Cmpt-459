# CMPT459 Data Mining
# Spring 2021 Milestone 1
# Lucia Schmidt & Joshua Peng
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# from src import step2_missingvals, step3_outliers, step4_transform
import step2_missingvals, step3_outliers, step4_transform, step5_join

def main():
    step2_missingvals.run_miss_va()
    # step3_outliers.remove_outliers()
    # step4_transform.transform_location_data()
    # step5_join.join_cases_with_locations()
    # df = pd.read_csv('../data/cases_train.csv')
    # test = df.iloc[22386]
    # print(test)
    # print(str(test['country']))
    # print(np.isnan(test['country']))

if __name__ == '__main__':
    main()

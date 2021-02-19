# CMPT459 Data Mining
# Spring 2021 Milestone 1
# Lucia Schmidt & Joshua Peng
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# from src import step2_missingvals, step3_outliers, step4_transform
import step2_missingvals, step3_outliers, step4_transform

def main():
    step2_missingvals.run_miss_va()
    step3_outliers.remove_outliers()
    #us_transformed_df = transform_location_data()
    step4_transform.transform_location_data()


if __name__ == '__main__':
    main()

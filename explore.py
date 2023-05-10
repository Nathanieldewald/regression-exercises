import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import wrangle as w
import scaling as s





def zillow_prepared(df):
    col_list = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'taxamount', 'tax_rate', 'price_per_sqft',
                'age']
    df_without_outliers = w.remove_outliers(df, 4, col_list)


    # split data
    col = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'taxamount', 'county', 'tax_rate',
           'price_per_sqft', 'age']
    col_scaled = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'taxamount', 'tax_rate', 'price_per_sqft',
                  'age']
    train, validate, test, train_scaled, validate_scaled, test_scaled = s.split_scale(df_without_outliers, col, col_scaled)
    return train, validate, test, train_scaled, validate_scaled, test_scaled
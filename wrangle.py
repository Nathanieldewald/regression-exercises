from sklearn.model_selection import train_test_split

from env import host, username, password
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def get_db_url(database):
    '''this function uses my env.py file to get the url to access the Codeup SQL server
     it takes the name of the database as an argument
     it returns the url'''
    database = database
    url = f'mysql+pymysql://{username}:{password}@{host}/{database}'
    return url


def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):

        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

def get_zillow_data():
    url = get_db_url('zillow')
    query = ''' select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
from properties_2017
where propertylandusetypeid = 261'''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)

    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe
    and return that dataframe with outliers removed'''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]  # filter dataframe
    return df

def train_validate_test(df, strat):
    '''
    This function will take in a dataframe and return train, validate, and test dataframes split
    where 55% is in train, 25% is in validate, and 20% is in test.
    '''
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=123,
                                            stratify=df[strat])
    train, validate = train_test_split(train_validate, test_size=0.25,
                                       random_state=123,
                                       stratify=train_validate[strat])
    return train, validate, test
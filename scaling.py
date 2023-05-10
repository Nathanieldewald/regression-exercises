from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import wrangle as w
# min max scaling
def min_max_scaling(train, validate, test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled 0-1
    """
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.transform(validate_scaled[columns])
    test_scaled[columns] = scaler.transform(test_scaled[columns])

    return train_scaled, validate_scaled, test_scaled

# standard scaling
def standard_scaling(train,validate,test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled to the mean and standard deviation
    """
    scaler = StandardScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.transform(validate_scaled[columns])
    test_scaled[columns] = scaler.transform(test_scaled[columns])
    return train_scaled, validate_scaled, test_scaled

# Robust Scaling
def robust_scaling(train,validate,test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled to the interquartile range
    """
    scaler = RobustScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.transform(validate_scaled[columns])
    test_scaled[columns] = scaler.transform(test_scaled[columns])
    return train_scaled, validate_scaled, test_scaled

# Scaling function
def scaling(train, validate, test, columns, method):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled
    """
    if method == 'min_max':
        return min_max_scaling(train, validate, test, columns)
    elif method == 'standard':
        return standard_scaling(train, validate, test, columns)
    elif method == 'robust':
        return robust_scaling(train, validate, test, columns)
    else:
        return train, validate, test

# Scaling function
def scale_inverse(train, validate, test, columns, method):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled
    """
    if method == 'min_max':
        return min_max_inverse(train, validate, test, columns)
    elif method == 'standard':
        return standard_inverse(train, validate, test, columns)
    elif method == 'robust':
        return robust_inverse(train, validate, test, columns)
    else:
        return train, validate, test

# min max scaling
def min_max_inverse(train, validate, test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled 0-1
    """
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.inverse_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.inverse_transform(validate_scaled[columns])
    test_scaled[columns] = scaler.inverse_transform(test_scaled[columns])

    return train_scaled, validate_scaled, test_scaled

# standard scaling
def standard_inverse(train,validate,test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled to the mean and standard deviation
    """
    scaler = StandardScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.inverse_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.inverse_transform(validate_scaled[columns])
    test_scaled[columns] = scaler.inverse_transform(test_scaled[columns])
    return train_scaled, validate_scaled, test_scaled

# Robust Scaling
def robust_inverse(train,validate,test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled to the interquartile range
    """
    scaler = RobustScaler()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.inverse_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.inverse_transform(validate_scaled[columns])
    test_scaled[columns] = scaler.inverse_transform(test_scaled[columns])
    return train_scaled, validate_scaled, test_scaled


def visualize_scaled_vs_unscaled(train, train_scaled):
    '''
    This function takes in the train and train_scaled dataframes and plots the distributions for each
    '''
    for i in train.columns:

        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.hist(train[i], bins=25, ec='black')
        plt.title(f'Original {i}')
        plt.subplot(122)
        plt.hist(train_scaled[i], bins=25, ec='black')
        plt.title(f'{i} scaled')
        plt.show()


# sklearns quantile transformer
def quantile_transformer(train, validate, test, columns):
    """
    This function takes in a dataframe and list of columns and returns the dataframe with the columns scaled
    """
    scaler = QuantileTransformer()
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    train_scaled[columns] = scaler.fit_transform(train_scaled[columns])
    validate_scaled[columns] = scaler.transform(validate_scaled[columns])
    test_scaled[columns] = scaler.transform(test_scaled[columns])
    return train_scaled, validate_scaled, test_scaled

# Function to split and scale data
def split_scale(df, col, col_scaled):
    '''
    This function takes in a dataframe
    splits into train, validate, and test
    then applies min max scaler to the data
    returns train, validate, test
    '''
    # split data
    train, validate, test = w.train_validate_test(df)
    # create scaler
    train_scaled, validate_scaled, test_scaled = min_max_scaling(train, validate, test, col_scaled)
    return train, validate, test, train_scaled, validate_scaled, test_scaled
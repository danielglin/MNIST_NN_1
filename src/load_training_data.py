# load_training_data.py
# Daniel Lin, November 2018
# This script provides functions for reading in the training data.

import numpy as np
import pandas as pd

def read_data(filepath):
    """
    This function reads in a csv containing examples in each row.
    
    Parameters:
        - filepath: the filepath of the csv containing the data
                The data is assumed to have the label in the first column.
        
    Returns:
        - X: a numpy array with the features of shape (nx, m), where
                m is the number of examples
                nx is the number of features
        - y: a numpy array with the labels of shape (nx, m), where
                m is the number of examples
                nx is the number of features
    """
    df = pd.read_csv(filepath, header=0)
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    y = y.values
    y = y.reshape(1, y.shape[0])
    X = X.values
    X = X/255.
    X = X.T
    
    # shuffle the data
    m = X.shape[1]
    rand_indices = np.random.permutation(m)
    X = X[:, rand_indices]
    y = y[:, rand_indices]
    #y = pd.get_dummies(y.flatten()).values.T
    # print('y', y)
    # print('y shape:', y.shape)
    # print('X shape:', X.shape)
    # print('X', X)
    return X, y
    
if __name__ == '__main__':
    X, Y = read_data('data/train.csv')
    print(Y)
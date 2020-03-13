import numpy as np
import quandl
import pandas as pd
#from params import *

quandl.ApiConfig.api_key = 'WnyxrQScej4rk7kPKFfy'

def get_data(stocks, col = "daily"):

    data = pd.DataFrame(columns= stocks, )
    for i in stocks:
        a = quandl.get(i,
                  start_date = '2000-12-31', end_date = '2019-01-01', transformation = 'rdiff'
                       , returns = 'numpy', collapse= col)
        data[i] = a['Close']

    return data

def get_full_quandl_exp(stocks, col = "daily"):
    
    data = pd.DataFrame(columns= stocks, )
    for i in stocks:

        a = quandl.get(i,
                  start_date = '2000-12-31', end_date = '2019-01-01', transformation = 'rdiff'
                       , returns = 'pandas', collapse= col)

    return a

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix,:], sequence[end_ix,:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



def train_test(X, y, batch_size):

    X = np.array(X)
    train_idx = int(X.shape[0] * 0.8)
    X_train = X[:train_idx, :]
    y_train = y[:train_idx, :]

    X_test = X[train_idx:,:]
    y_test = y[train_idx:, :]


   # X_train, X_val, X_test = reshaper_X(X_train), reshaper_X(X_val), reshaper_X(X_test)
    y_train, y_test = y_train.reshape(y_train.shape[0], y_train.shape[1], 1), y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    return X_train, X_test, y_train, y_test


def get_batch(source, i, is_target = True):
    if is_target:
        seq_len = min(args_batch_size, source.shape[0] - 1 - i)
        data = source[i : i + seq_len,:,:]
    else:
        seq_len = min(args_batch_size, source.shape[0] - 1 - i)
        data = source[i : i + seq_len,:,:]
    return data

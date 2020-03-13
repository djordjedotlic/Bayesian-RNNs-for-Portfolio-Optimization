from data_v2 import *
import numpy as np
from mvar import *

def historical_portfolio_fun(stocks, n_steps, batch_size, col = 'daily', gamma = 1):
    mu_hist = []
    var_hist = []
    
    n_stocks = len(stocks)
    print(n_stocks)
    
    df =np.array(get_data(stocks= stocks, col = col))
    X,y = split_sequence(df, n_steps)
    X_train, X_test, y_train, y_test = train_test(X,y, batch_size= batch_size)

    w0 = [1/X_test.shape[2]] * X_test.shape[2]
    
    model = 'historical'
    estimation = 'deterministic'

    #Calculate mean and variance
    for tstamp in range(X_test.shape[0]):
        mu_hist.append(np.asmatrix(np.mean(X_test[tstamp, :, :], axis = 0)).reshape(n_stocks,1))
        var_hist.append(np.asmatrix(np.cov(X_test[tstamp, :, :], rowvar = False)))

    #caluclate optimal portfolio weights given gamma    

    historical_weights = []    
    for ts in range(len(mu_hist)):
        hv = (mv_portfolio(returns = mu_hist[ts], variance = var_hist[ts], mv_obj = mv_obj, w0 = w0, gamma = gamma))
        if not any(np.isnan(hv)):
            historical_weights.append(hv)
        else:
            historical_weights.append(historical_weights[ts - 1])     
        
        
    #Predict the portfolio returns, and calculate expected returns and variance of portfolio

    value_per_position = [None] #* len(gamma) 
    portfolio_value = [None] #* len(gamma)
    
    value_per_position = [historical_weights[0] + historical_weights[0] * y_test[0].T]
    portfolio_value = [sum(value_per_position[0].T) - 1]

    for n in range(1,len(historical_weights)):
        value_per_position.append(value_per_position[n-1] + historical_weights[n] * y_test[n].T)
        portfolio_value.append(sum((value_per_position[n].T)) - 1)


        
    return np.asarray(portfolio_value)
    
    
    
def equal_weights_fun(stocks, n_steps, batch_size, col = 'daily'):
    
    
    df =np.array(get_data(stocks= stocks, col = col))
    X,y = split_sequence(df, n_steps)
    X_train, X_test, y_train, y_test = train_test(X,y, batch_size= batch_size)

    w0 = [1/X_test.shape[2]] * X_test.shape[2]


    #caluclate optimal portfolio weights given gamma    

    historical_weights = []    
    for ts in range(len(y_test)):
        historical_weights.append(w0)
        
    #Predict the portfolio returns, and calculate expected returns and variance of portfolio

    value_per_position = [None] #* len(gamma) 
    portfolio_value = [None] #* len(gamma)
    
    value_per_position = [historical_weights[0] + historical_weights[0] * y_test[0].T]
    portfolio_value = [sum(value_per_position[0].T) - 1]

    for n in range(1,len(historical_weights)):
        value_per_position.append(value_per_position[n-1] + historical_weights[n] * y_test[n].T)
        portfolio_value.append(sum((value_per_position[n].T)) - 1)


        
    return np.asarray(portfolio_value)
from scipy.optimize import minimize
import numpy as np


def mv_obj(w, mu, sigma, gamma = 1):
    w = np.asmatrix(w)
    return  - w*mu + (gamma/2)*w*sigma*w.T

def mv_portfolio(returns, variance, mv_obj, w0, gamma):

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1.0})
    bnds = [(0,1)] * returns.shape[0]
    options = {'ftol':1e-20, 'maxiter':2000, 'disp':False}

    res = minimize(mv_obj, w0, args = (returns, variance, gamma), method='SLSQP',
                       bounds=bnds, constraints=cons, options=options)

    return res.x

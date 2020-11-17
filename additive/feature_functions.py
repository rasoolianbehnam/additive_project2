from additive.utility import *
from scipy.stats import mode
import numpy as np
feature_functions_functions = {}
def feature(fun):
    feature_functions_functions[fun.__name__] = fun
    return fun

def ra_1d_(x):
    return np.mean(np.abs(x-np.mean(x, axis=1, keepdims=True)), axis=1)

def rq_1d_(x):
    return np.sqrt(np.mean((x-np.mean(x, axis=1, keepdims=True))**2, axis=1))

@feature
def ra_1d(x):
    return ra_1d_(x).mean()

@feature
def rq_1d(x):
    return rq_1d_(x).mean()

@feature
def rsk_1d(x):
    rq = rq_1d_(x)
    return (np.mean((x-np.mean(x, axis=1, keepdims=True))**3, axis=1)/rq**3).mean()

@feature
def rku_1d(x):
    rq = rq_1d_(x)
    return (np.mean((x-np.mean(x, axis=1, keepdims=True))**4, axis=1)/rq**4).mean()

@feature
def rp_1d(x):
    return np.max(x-np.mean(x, axis=1, keepdims=True), axis=1).mean()

@feature
def rv_1d(x):
    return -np.min(x - np.mean(x, axis=1, keepdims=True), axis=1).mean()

@feature
def ra_2d(x):
    return np.mean(np.abs(x-np.mean(x)))

@feature
def rq_2d(x):
    return np.sqrt(np.mean((x-np.mean(x))**2))

@feature
def rp_2d(x):
    return np.max(x-np.mean(x))

@feature
def rv_2d(x):
    return -np.min(x - np.mean(x))

@feature
def rsk_2d(x):
    rq = rq_2d(x)
    return np.mean((x-np.mean(x))**3)/rq**3

@feature
def rku_2d(x):
    rq = rq_2d(x)
    return np.mean((x-np.mean(x))**4)/rq**4

@feature
def mode_1d(x):
    modes = mode((x-x.mean()).round(decimals=0), axis=1)
    return modes.mode.mean()

@feature
def mode_2d(x, bins=100):
    freq, val = np.histogram(x-x.mean(), bins=bins, density=True, )
    return val[np.argmax(freq)]


feature(np.median)
feature(np.mean)
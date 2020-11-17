import numpy as np
import pandas as pd
import cv2
import pickle
import collections
import seaborn as sns
import glob
from additive.experimental import *
from matplotlib.pyplot import *
from additive.utility import *
from os.path import basename, splitext
import dask
from dask import bag, compute, delayed
from dask.distributed import Client
from additive.experimental import extract_circles

Circle = collections.namedtuple('Circle', 'beg, end, index, h, cx, cy, r, x_mean,  x_std, y_mean, y_std')

columns = ['profile'] + list(Circle._fields)
feature_functions = set()
circle_functions = set()
def feature(fun):
    feature_functions.add(fun.__name__)
    return fun

def circle_feature(fun):
    circle_functions.add(fun.__name__)
    return fun

def log(s):
    print("[*] %s"%s)
    
def p(n):
    def pp(x):
        return np.percentile(x, n)
    return pp

profile, beg, end, index, h, cx, cy, r =  'profile beg end index h cx cy r'.split()
#stats = df[[profile, r]].groupby(profile).aggregate(['mean', 'median', 'std', p(5)])
#smalles_5percent = df.apply(lambda x: x.r < stats.r.pp.loc[x.profile], axis=1)

class Features:
    circle_statistics = None
    statistics = None
    circles = None
    exceptions = []
    def __init__(self, x, edge_to_ignore=0):
        log('Starting...')
        if edge_to_ignore > 0:
            x = x[edge_to_ignore:-edge_to_ignore, edge_to_ignore:-edge_to_ignore]
        self.x = np.ma.array(x, mask=x<x.mean(axis=1, keepdims=True)-3.5*x.std(axis=1, keepdims=True))
        #self.x = np.array()#, mask=x<x.mean(axis=1, keepdims=True)-3.5*x.std(axis=1, keepdims=True))
        #y = self.x[self.x>self.x.mean(axis=1, keepdims=True)-3.5*self.x.std(axis=1, keepdims=True)]
        #self.kernel_sizes = [int(np.percentile(get_cut_points(profile, profile.mean()), 25)) for profile in self.x]
        self.kernel_sizes = [int(np.percentile(get_cut_points(profile, np.ma.median(profile)), 25)) 
                             for profile in self.x]
        log('Kernel sizes extracted')
        self.local_minima = [get_local_minima_2d(profile, kernel_size, mn=-1) 
                             for profile, kernel_size in zip(self.x, self.kernel_sizes)]
        log('Local minima extracted')
        self.local_maxima = [get_local_minima_2d(-profile, kernel_size) 
                             for profile, kernel_size in zip(self.x, self.kernel_sizes)]
        log('Local maxima extracted')
 
    @feature
    def ra(self):
        return np.mean(np.abs(self.x-np.mean(self.x, axis=1, keepdims=True)), axis=1)

    @feature
    def rq(self):
        return np.sqrt(np.mean((self.x-np.mean(self.x, axis=1, keepdims=True))**2, axis=1))


    @feature
    def rv(self):
        return np.max(self.x-np.mean(self.x, axis=1, keepdims=True), axis=1)

    @feature
    def rz(self):
        return -np.min(self.x - np.mean(self.x, axis=1, keepdims=True), axis=1)

    @feature
    def r10_iso(self):
        return np.array([np.mean(np.sort(self.x[n][self.local_maxima[n]])[:10] ) for n in range(len(self.x))]) - \
            np.array([np.mean(np.sort(self.x[n][self.local_minima[n]])[-10:]) for n in range(len(self.x))])
    
    def get_all_circles(self):
        allCircles = []
        for i, profile in enumerate(self.x):
            try:
                x, y, der1, der2, local_minima_2d, circles = extract_circles(np.arange(len(profile)), 
                                                                     profile,
                                                                     kernel_size_param=self.kernel_sizes[i],
                                                                     same_scale=True, verbose=False)       
                if i%500==0:
                    log("Finished extracting circles from profile %5d/%5d"%(i, len(self.x)))
                allCircles.append(circles)
            except Exception as e:
                self.exceptions.append(["Exception occurred for profile %d"])
        self.circles = pd.DataFrame(((i, *circle) for i, circles in enumerate(allCircles) for circle in circles), columns=columns)   
        return self
    
    @circle_feature
    def rho(self):
        if self.circles is None:
            self.get_all_circles()
        return self.circles[['profile', 'r']].groupby('profile').aggregate('mean')
    
    @circle_feature
    def rho95(self):
        if self.circles is None:
            self.get_all_circles()       
        mardas = self.circles[['profile', 'r']].groupby('profile').aggregate(p(5))
        return self.circles[self.circles.apply(lambda x: x.r < mardas.r.loc[x.profile], axis=1)][['profile', 'r']].groupby('profile').aggregate('mean')
       
    def run_all_tests(self):
        log('Starting to run all tests.')
        self.run_all_rhos()
        return self.run_all_non_rhos()
    
    def run_all_non_rhos(self):
        log('Starting to get global properties')
        if self.statistics is None:
            self.statistics = pd.DataFrame({name: getattr(self, name)() for name in feature_functions})
        return self
    
    def run_all_rhos(self):
        log('Starting to get global properties')
        if self.circle_statistics is None:
            circle_statistics = {name: getattr(self, name)() for name in circle_functions}
            self.circle_statistics = pd.concat((x for x in circle_statistics.values()), axis=1)
            self.circle_statistics.columns = circle_statistics.keys()
    #self.circle_statistics = {name: getattr(self, name)() for name in circle_functions}
        return self
        

#GlobalFeatures = collections.namedtuple('GlobalFeatures', feature_functions)

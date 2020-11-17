#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

import numpy as np
import pandas as pd
import cv2
import pickle
import collections
import seaborn as sns
import glob
from matplotlib.pyplot import *
from additive.utility import *
from pathos.multiprocessing import ProcessingPool, ThreadingPool
from os.path import basename, splitext



columns = ['profile'] + list(Circle._fields)
functions = set()
circle_functions = set()
def feature(fun):
    functions.add(fun.__name__)
    return fun

def circle_feature(fun):
    circle_functions.add(fun.__name__)
    return fun

def log(s):
    print("[*] %s"%s)
    
def percentile(n):
    def pp(x):
        return np.percentile(x, n)
    return pp

profile, beg, end, index, h, cx, cy, r =  'profile beg end index h cx cy r'.split()
#stats = df[[profile, r]].groupby(profile).aggregate(['mean', 'median', 'std', percentile(5)])
#smalles_5percent = df.apply(lambda x: x.r < stats.r.pp.loc[x.profile], axis=1)

class Features:
    circle_statistics = None
    statistics = None
    circles = None
    exceptions = []
    def __init__(self, x):
        log('Starting...')
        self.x = np.ma.array(x, mask=x<x.mean(axis=1, keepdims=True)-3*x.std(axis=1, keepdims=True))
        self.kernel_sizes = [int(np.percentile(get_cut_points(profile, profile.mean()), 25)) for profile in self.x]
        log('Kernel sizes extracted')
        self.local_minima = [get_local_minima_2d(profile, kernel_size, mn=-1) for profile, kernel_size in zip(self.x, self.kernel_sizes)]
        log('Local minima extracted')
        self.local_maxima = [get_local_minima_2d(-profile, kernel_size) for profile, kernel_size in zip(self.x, self.kernel_sizes)]
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
        return np.array([np.mean(np.sort(self.x[n][self.local_maxima[n]])[:10] ) for n in range(len(self.x))]) -             np.array([np.mean(np.sort(self.x[n][self.local_minima[n]])[-10:]) for n in range(len(self.x))])
    
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
        mardas = self.circles[['profile', 'r']].groupby('profile').aggregate(percentile(5))
        return self.circles[self.circles.apply(lambda x: x.r < mardas.r.loc[x.profile], axis=1)][['profile', 'r']].groupby('profile').aggregate('mean')
       
    def run_all_tests(self):
        log('Starting to run all tests.')
        self.run_all_rhos()
        return self.run_all_non_rhos()
    
    def run_all_non_rhos(self):
        log('Starting to get global properties')
        if self.statistics is None:
            self.statistics = pd.DataFrame({name: getattr(self, name)() for name in functions})
        return self
    
    def run_all_rhos(self):
        log('Starting to get global properties')
        if self.circle_statistics is None:
            circle_statistics = {name: getattr(self, name)() for name in circle_functions}
        self.circle_statistics = pd.concat((x for x in circle_statistics.values()), axis=1)
        self.circle_statistics.columns = circle_statistics.keys()
    #self.circle_statistics = {name: getattr(self, name)() for name in circle_functions}
        return self
        

#GlobalFeatures = collections.namedtuple('GlobalFeatures', functions)

def random_sub_image(img, ratio=.5):
    w, h = img.shape
    ww, hh = int(ratio*w), int(ratio*h)
    x, y = np.random.randint(max(1,w-ww)), np.random.randint(max(1,h-hh))
    return img[x:x+ww, y:y+hh]

def random_sub_length_image(img, ratio=.5):
    w, h = img.shape
    ww, hh = int(w), int(ratio*h)
    x, y = 0, np.random.randint(max(1,h-hh))
    #print(x, y)
    return img[:, y:y+hh]

def random_sub_width_image(img, ratio=.5):
    w, h = img.shape
    ww, hh = int(w*h), int(ratio)
    x, y = np.random.randint(max(1,w-ww)), 0
    return img[x:x+ww, :]


# In[6]:


images_mapper = {
          'Normal': lambda x:x, 
          'Smoothed': lambda x: cv2.pyrUp(cv2.pyrDown((x))),
#          '50% area rotate': lambda x: align_image(random_sub_image(x)),
#          '50% area tilt': lambda x: adjust_tilt(random_sub_image(x)),
#          '25% area': lambda x: random_sub_image(x, .05),
#          '50% length': lambda x: random_sub_length_image(x),
#          '50% length rotate': lambda x: align_image(random_sub_length_image(x)),
#          '50% length tilt': lambda x: adjust_tilt(random_sub_length_image(x)),
#          '50% width': lambda x: random_sub_width_image(x),
#          '50% width tilt': lambda x: adjust_tilt(random_sub_width_image(x)),
#          'Rotated': lambda x: align_image(x),
#          'Tilted': lambda x: adjust_tilt(x),
#          'Tilted & Rotated': lambda x: adjust_tilt(align_image(x))
                }


# In[10]:


file_names = glob.glob('dataset/1000x/*_2.np')
file_names
#with ThreadingPool(2) as t:
#    stats_list = t.map(get_all_stats, file_names)
#stats_list = get_all_stats(file_names[0])


# In[11]:


def foo(args):
    (k, f), x = args
    return (k, f(x))
def get_all_stats(file_name):
    with open(file_name, 'rb') as f:
        img = pickle.load(f)
    images = dict(foo(x) for x in zip(images_mapper.items(), [img]*len(images_mapper)))
    with ProcessingPool() as pool:
        images_features = pool.map(Features, images.values())
    with ProcessingPool() as pool:
        images_features = pool.map(lambda x: x.run_all_tests(), images_features)
    return images_features
with ThreadingPool(4) as t:
    result = t.map(get_all_stats, file_names)


# In[62]:


#stats = {name: feature[0].statistics for name, feature in zip(file_names, result)}
stats = {f_name:{k:feature.statistics for k, feature in zip(images_mapper, features)} for f_name, features in zip(file_names, result)}


# In[70]:


#circle_stats = {name: mardas(feature[0]) for name, feature in zip(file_names, result)}
#circle_stats = {name: feature[0].circle_statistics for name, feature in zip(file_names, result)}
circle_stats = {f_name:{k:feature.circle_statistics for k, feature in zip(images_mapper, features)} for f_name, features in zip(file_names, result)}


# In[3]:


with open('dataset/1000x/comparison_local_stats.pkl', 'rb') as f:
    circle_stats = pickle.load(f)
with open('dataset/1000x/comparison_global_stats.pkl', 'rb') as f:
    stats = pickle.load(f)   


# In[7]:


name_dict = {'ra':"$R_a$", 'rq':"$R_q$", 'rv':"$R_v$", 'rz':"$R_z$", 'r10_iso':"$R_{10(ISO)}$"}
names = images_mapper.keys() - set('50% area rotate; 50% area tilt; 50% length tilt; 50% width tilt; 50% length rotate; 50% length tilt'.split('; '))
statistics = ['mean']
data = pd.DataFrame(
    [(file_name, name_dict[column], name, statistic, stat[name][column].describe()[statistic])
     for file_name, stat in stats.items()
     for i,name in enumerate(names) 
     for column in stat[name].columns
     for statistic in statistics],
    columns=["File Name", 'Measure', 'Variation', 'Statistics', 'Statistic Value($\mu m$)'])


# In[8]:


data['File Name'] = data['File Name'].apply(lambda x: x.split('/')[-1].split('_')[0][1:]+'x')


# In[9]:


data


# In[18]:


sns.set_palette("deep")
ax = sns.catplot(data=data[data['Variation']=='Normal'], kind='bar',  height=8.27, aspect=11.7/8.27, x='Measure', y='Statistic Value($\mu m$)', hue='File Name', row='Statistics')
savefig('global_measure_comparison_500x_vs_1000x.svg')


# In[19]:


sns.set_palette("deep")
ax = sns.catplot(data=data, kind='bar',  height=8.27, aspect=11.7/8.27, x='Measure', y='Statistic Value($\mu m$)', hue='Variation', row='Statistics')
savefig('global_measure_comparison_normal_vs_smoothed.svg')


# In[12]:


name_dict = {'ra':"$R_a$", 'rq':"$R_q$", 'rv':"$R_v$", 'rz':"$R_z$", 'r10_iso':"$R_{10(ISO)}$", 'rho':r"$\rho$", 'rho95': r"$\rho_{95}$"}
names = images_mapper.keys() - set('50% area rotate; 50% area tilt; 50% length tilt; 50% width tilt; 50% length rotate; 50% length tilt'.split('; '))
statistics = ['mean']
data = pd.DataFrame(
    [(file_name, name_dict[column], name, statistic, stat[name][column].describe()[statistic])
     for file_name, stat in circle_stats.items()
     for i,name in enumerate(names) 
     for column in stat[name].columns
     for statistic in statistics],
    columns=["File Name", 'Measure', 'Variation', 'Statistics', 'Statistic Value($\mu m$)'])


# In[13]:


data['File Name'] = data['File Name'].apply(lambda x: x.split('/')[-1].split('_')[0][1:]+'x')


# In[22]:


sns.set_palette("deep")
ax = sns.catplot(data=data[data['Variation']=='Normal'], kind='bar',  height=8.27, aspect=11.7/8.27, x='Measure', y='Statistic Value($\mu m$)', hue='File Name', row='Variation')
savefig('local_measure_comparison_500x_vs_1000x.svg')


# In[23]:


sns.set_palette("deep")
ax = sns.catplot(data=data, kind='bar',  height=8.27, aspect=11.7/8.27, x='Measure', y='Statistic Value($\mu m$)', hue='Variation')
savefig('local_measure_comparison_normal_vs_smoothed.svg')


# In[ ]:





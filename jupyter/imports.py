import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import optimize
from scipy import sqrt
from matplotlib.patches import Ellipse
from matplotlib.pyplot import *
from sklearn.linear_model import LinearRegression
import collections
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import pickle
import collections
import seaborn as sns
import glob
from matplotlib.pyplot import *
from os.path import basename, splitext
import dask
from dask import bag, compute, delayed
from dask.distributed import Client
import joblib
from multiprocessing import Pool

from collections import namedtuple
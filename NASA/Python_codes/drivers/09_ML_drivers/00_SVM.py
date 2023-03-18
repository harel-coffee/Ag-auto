####
#### June 1, 2022
####


import numpy as np
import pandas as pd
import scipy, scipy.signal


import datetime
from datetime import date
import time


from tslearn.metrics import dtw as dtw_metric
from dtaidistance import dtw # https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw_visualisation as dtwvis

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

from random import seed
from random import random

import sys
import os, os.path
import shutil

import h5py
import pickle

start_time = time.time()

# search path for modules# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

# sys.path.append('/home/hnoorazar/NASA/')
# import NASA_core as nc
# import NASA_plot_core as ncp

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
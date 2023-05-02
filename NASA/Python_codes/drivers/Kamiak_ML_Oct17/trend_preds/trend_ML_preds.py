import shutup

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
import time

import random
from random import seed, random

import os, os.path, shutil

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# import matplotlib
# import matplotlib.pyplot as plt
# from pylab import imshow

import scipy, scipy.signal
import pickle, h5py
import sys

from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc


try:
    print("numpy.__version__=", numpy.__version__)
except:
    print("umpy.__version__ not printed")

####################################################################################
###
###      Parameters
###
####################################################################################

indeks = sys.argv[1]
smooth = sys.argv[2]
batch = str(sys.argv[3])
ML_model = sys.argv[4]

####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

if ML_model != "DL":
    if smooth_type == "regular":
        in_dir = data_base + "VI_TS/04_regularized_TS/"
    else:
        in_dir = data_base + "VI_TS/05_SG_TS/"
else:
    in_dir = data_base + "06_cleanPlots_4_DL/" + indeks + "_" + smooth + "plots/"

out_dir = data_base + "ML_preds/" + indeks + "_" + smooth_type + "_" + ML_model + "/"
os.makedirs(out_dir, exist_ok=True)


#####################################################################
######
######                           Body
######
#####################################################################
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img


####################################################################################
###
###      Read
###
####################################################################################

winnerModels = pd.read_csv(param_dir + "winnerModels.csv")

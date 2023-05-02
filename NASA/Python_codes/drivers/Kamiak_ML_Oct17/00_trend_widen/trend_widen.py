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

VI_idx = sys.argv[1]
smooth = sys.argv[2]
batch_no = str(sys.argv[3])

####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"
if smooth_type == "regular":
    in_dir = data_base + "VI_TS/04_regularized_TS/"
else:
    in_dir = data_base + "VI_TS/05_SG_TS/"

out_dir = data_base + VI_idx + "_" + smooth_type + "_" + "/"
os.makedirs(out_dir, exist_ok=True)


#####################################################################
######
######                           Body
######
#####################################################################
f_name = VI_idx + "_" + smooth + "_" + "intersect_batchNumber" + batch_no + "_JFD.csv"
data = pd.read_csv(in_dir + f_name)
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])

##############################
##
##     Widen
##
##############################
#
# Form an empty dataframe to populate
#
VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID"] + VI_colnames

years = data.human_system_start_time.dt.year.unique()
IDs = data.ID.unique()
no_rows = len(IDs) * len(years)

data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
data_wide.ID = list(IDs) * len(years)
data_wide.sort_values(by=["ID"], inplace=True)
data_wide.reset_index(drop=True, inplace=True)
data_wide.year = list(years) * len(IDs)


for an_ID in IDs:
    curr_field = data[data.ID == an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]

        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

        if VI_idx == "EVI":
            data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
        elif VI_idx == "NDVI":
            data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

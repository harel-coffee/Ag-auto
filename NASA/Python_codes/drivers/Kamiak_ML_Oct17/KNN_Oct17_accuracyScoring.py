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
smooth_type = sys.argv[2]
SR = int(sys.argv[3])

print(f"Passed Args. are: {VI_idx=:}, {smooth_type=:}, and {SR=:}!")

"""
    # Metadata
"""
NASA_dir_base = "/data/project/agaid/h.noorazar/NASA/"
meta_dir = NASA_dir_base + "parameters/"

meta = pd.read_csv(meta_dir + "evaluation_set.csv")
meta_moreThan10Acr = meta[meta.ExctAcr > 10]
print(meta.shape)
print(meta_moreThan10Acr.shape)

"""
   # Read Training Set Labels
"""
ML_data_folder = NASA_dir_base + "ML_data_Oct17/"
overSamples_data_folder = NASA_dir_base + "ML_data_Oct17/overSamples/"

model_dir = NASA_dir_base + "ML_Models_Oct17/overSample/"
os.makedirs(model_dir, exist_ok=True)


def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = (
    VI_idx
    + "_"
    + smooth_type
    + "_wide_train80_split_2Bconsistent_Oct17_overSample"
    + str(SR)
    + ".csv"
)
wide_overSample = pd.read_csv(overSamples_data_folder + f_name)  # train80_GT_wide:

# wide_overSample = wide_overSample.loc[1:30].copy() #### For testing the code. Comment out after testing the code.
print("train set size of sample ratio " + str(SR) + " is", wide_overSample.shape)

x_train_df = wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df = wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
assert list(x_train_df.ID) == list(y_train_df.ID)

parameters = {
    "n_neighbors": [2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
    "weights": ["uniform", "distance"],
}
KNN_DTW_prune = GridSearchCV(KNeighborsClassifier(metric=DTW_prune), parameters, cv=5, verbose=1)
KNN_DTW_prune.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

modelOutName = (
    model_dir
    + "KNN_"
    + smooth_type
    + "_"
    + VI_idx
    + "_Oct17_AccScoring_Oversample_SR"
    + str(SR)
    + ".sav"
)
pickle.dump(KNN_DTW_prune, open(modelOutName, "wb"))
print("dumped")
try:
    print("KNN_DTW_prune.best_params_ is:")
    print(KNN_DTW_prune.best_params_)
except:
    print("An exception occurred")
try:
    print("KNN_DTW_prune.best_score_")
    print(KNN_DTW_prune.best_score_)
except:
    print("An exception occurred 2")

##########
##########    Test
##########
x_test_df = pd.read_csv(
    ML_data_folder
    + "widen_test_TS/"
    + VI_idx
    + "_"
    + smooth_type
    + "_wide_test20_split_2Bconsistent_Oct17.csv"
)

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

KNN_DTW_prune_predictions = KNN_DTW_prune.predict(x_test_df.iloc[:, 1:])
KNN_DTW_prune_y_test_df = y_test_df.copy()
KNN_DTW_prune_y_test_df["prediction"] = list(KNN_DTW_prune_predictions)


test_result_dir = NASA_dir_base + "ML_test_result_Oct17/overSample/"
os.makedirs(test_result_dir, exist_ok=True)

out_name = (
    test_result_dir
    + "KNN_"
    + smooth_type
    + "_"
    + VI_idx
    + "_Oct17_AccScoring_Oversample_SR"
    + str(SR)
    + "_test_result.csv"
)
KNN_DTW_prune_y_test_df.to_csv(out_name, index=False)
# print ()
# print (KNN_DTW_prune_y_test_df.head(2))
# print ()

true_single_predicted_single = 0
true_single_predicted_double = 0

true_double_predicted_single = 0
true_double_predicted_double = 0

for index_ in KNN_DTW_prune_y_test_df.index:
    curr_vote = list(KNN_DTW_prune_y_test_df[KNN_DTW_prune_y_test_df.index == index_].Vote)[0]
    curr_predict = list(
        KNN_DTW_prune_y_test_df[KNN_DTW_prune_y_test_df.index == index_].prediction
    )[0]
    if curr_vote == curr_predict:
        if curr_vote == 1:
            true_single_predicted_single += 1
        else:
            true_double_predicted_double += 1
    else:
        if curr_vote == 1:
            true_single_predicted_double += 1
        else:
            true_double_predicted_single += 1

KNN_DTW_prune_confus_tbl_test = pd.DataFrame(
    columns=["None", "Predict_Single", "Predict_Double"], index=range(2)
)
KNN_DTW_prune_confus_tbl_test.loc[0, "None"] = "Actual_Single"
KNN_DTW_prune_confus_tbl_test.loc[1, "None"] = "Actual_Double"
KNN_DTW_prune_confus_tbl_test["Predict_Single"] = 0
KNN_DTW_prune_confus_tbl_test["Predict_Double"] = 0

KNN_DTW_prune_confus_tbl_test.loc[0, "Predict_Single"] = true_single_predicted_single
KNN_DTW_prune_confus_tbl_test.loc[0, "Predict_Double"] = true_single_predicted_double
KNN_DTW_prune_confus_tbl_test.loc[1, "Predict_Single"] = true_double_predicted_single
KNN_DTW_prune_confus_tbl_test.loc[1, "Predict_Double"] = true_double_predicted_double
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("KNN_DTW_prune_confus_tbl_test")
print(KNN_DTW_prune_confus_tbl_test)

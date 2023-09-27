import shutup  # , random

shutup.please()

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd
from datetime import date, datetime
from random import seed, random

import sys, os, os.path, shutil, h5py, time, pickle
import matplotlib
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

# sys.path.append("/home/h.noorazar/NASA/")
# import NASA_core as nc

####################################################################################
###
###      Parameters
###
####################################################################################

VI_idx = sys.argv[1]
smooth = sys.argv[2]
train_ID = sys.argv[3]  # we have different training sets: 1, 2, 3, 4, 5, 6
SR = sys.argv[4]  # sample Ratio 3, 4, 5, 6, 7, 8
ML_model = sys.argv[5]
print("Passed Args. are: ", VI_idx, ",", smooth, ",", train_ID, ",", SR)
####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

ML_data_dir_base = data_base + "/ML_data_Oct17/"
overSamp_data_base = ML_data_dir_base + "overSamples/"

model_dir = data_base + "ML_Models_Oct17/DeskReject/"
os.makedirs(model_dir, exist_ok=True)

train_test_dir = overSamp_data_base + "train_test_DL_" + str(train_ID) + "/"
train_plot_dir = train_test_dir + "/oversample" + str(SR) + "/" + smooth + "_" + VI_idx + "_train/"
train_test_split_dir = ML_data_dir_base + "train_test_DL_" + str(train_ID) + "/"


#####################################################################
######
######                           Body
######
#####################################################################
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


LL = "_wide_train80_split_2Bconsistent_Oct17_overSample"
train_fileName = VI_idx + "_" + smooth + LL + str(SR) + ".csv"
train80_wide = pd.read_csv(train_test_dir + train_fileName)
print(train80_wide.shape)

x_train_df = train80_wide.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df = train80_wide[["ID", "Vote"]]

print("===================================================")

if ML_model == "KNN":
    parameters = {
        "n_neighbors": [2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
        "weights": ["uniform", "distance"],
    }
    KNN_DTW_prune = GridSearchCV(
        KNeighborsClassifier(metric=DTW_prune), parameters, cv=5, verbose=1
    )
    KNN_DTW_prune.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

    modelOutName = (
        model_dir
        + "KNN_"
        + smooth
        + "_"
        + VI_idx
        + "_train_ID"
        + str(train_ID)
        + "_Desk_AccScor_SR"
        + str(SR)
        + ".sav"
    )
    pickle.dump(KNN_DTW_prune, open(modelOutName, "wb"))
    print("dumped")
elif ML_model == "RF":
    parameters = {
        "n_jobs": [6],
        "criterion": ["gini", "entropy"],  # log_loss
        "max_depth": [2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
        "min_samples_split": [2, 3, 4, 5],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample", None],
        "ccp_alpha": [0.0, 1, 2, 3],
        # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
        "max_samples": [None, 1, 2, 3, 4, 5],
    }

    RF_grid_2 = GridSearchCV(
        RandomForestClassifier(random_state=0), parameters, cv=5, verbose=1, error_score="raise"
    )
    RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

    modelOutName = (
        model_dir
        + ML_model
        + "_"
        + VI_idx
        + "_"
        + smooth
        + "_grid_2_Desk_AccScor_SR"
        + str(SR)
        + "_train_ID"
        + str(train_ID)
        + ".sav"
    )
    pickle.dump(RF_grid_2, open(modelOutName, "wb"))

    print(RF_grid_2.best_params_)
    print(RF_grid_2.best_score_)


elif ML_model == "SVM":
    parameters = {
        "C": [5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
        "kernel": [
            "linear",
            "poly",
            "rbf",
            "sigmoid",
        ],  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    }  # ,
    SVM_classifier_NoneWeight_00 = GridSearchCV(SVC(random_state=0), parameters, cv=5, verbose=1)
    SVM_classifier_NoneWeight_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

    modelOutName = (
        model_dir
        + ML_model
        + "_"
        + VI_idx
        + "_"
        + smooth
        + "_NoneWeight_00_Desk_AccScor_SR"
        + str(SR)
        + "_train_ID"
        + str(train_ID)
        + ".sav"
    )
    pickle.dump(SVM_classifier_NoneWeight_00, open(modelOutName, "wb"))

    print(SVM_classifier_NoneWeight_00.best_params_)
    print(SVM_classifier_NoneWeight_00.best_score_)

    SVM_classifier_balanced_00 = GridSearchCV(
        SVC(random_state=0, class_weight="balanced"), parameters, cv=5, verbose=1
    )
    SVM_classifier_balanced_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)
    modelOutName = (
        model_dir
        + ML_model
        + "_"
        + VI_idx
        + "_"
        + smooth
        + "_balanced_Desk_AccScor_SR"
        + str(SR)
        + "_train_ID"
        + str(train_ID)
        + ".sav"
    )
    pickle.dump(SVM_classifier_balanced_00, open(modelOutName, "wb"))
    print(SVM_classifier_balanced_00.best_params_)
    print(SVM_classifier_balanced_00.best_score_)


##############################################################################

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

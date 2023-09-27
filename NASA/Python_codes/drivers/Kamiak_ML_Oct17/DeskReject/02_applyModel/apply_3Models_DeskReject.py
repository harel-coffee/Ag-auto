import shutup  # , random

shutup.please()

# from keras.optimizers import SGD
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd
from datetime import date, datetime
from random import seed, random
import sys, os, os.path, shutil, h5py, time, pickle

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

####################################################################################
###
###      Parameters
###
####################################################################################

VI_idx = sys.argv[1]
smooth = sys.argv[2]
train_ID = sys.argv[3]
SR = sys.argv[4]  # sample Ratio 3, 4, 5, 6, 7, 8
ML_model = sys.argv[5]

print("Passed Args. are: ", VI_idx, ",", smooth, ",", ML_model, ",", SR)
####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

ML_data_dir_base = data_base + "/ML_data_Oct17/"
overSamp_data_base = ML_data_dir_base + "overSamples/"
model_dir = data_base + "ML_Models_Oct17/DeskReject/"

train_test_dir = overSamp_data_base + "train_test_DL_" + str(train_ID) + "/"
train_plot_dir = train_test_dir + "/oversample" + str(SR) + "/" + smooth + "_" + VI_idx + "_train/"
train_test_split_dir = ML_data_dir_base + "train_test_DL_" + str(train_ID) + "/"

res_dir = data_base + "DeskRejectResults/00_predictions/"
os.makedirs(res_dir, exist_ok=True)


#####################################################################
######
######                           Body
######
#####################################################################
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


LL = "_wide_train80_split_2Bconsistent_Oct17_overSample"
train_fileName = VI_idx + "_" + smooth + LL + str(SR) + ".csv"
train80_wide = pd.read_csv(train_test_dir + train_fileName)
train80_wide.drop_duplicates(inplace=True)
LL = "_wide_test20_split_2Bconsistent_Oct17"
test_fileName = VI_idx + "_" + smooth + LL + ".csv"
test20_wide = pd.read_csv(train_test_dir + test_fileName)


print(test20_wide.shape)
print(train80_wide.shape)
print("===================================================")

if ML_model == "KNN":
    model_name = (
        "KNN_" + smooth + "_" + VI_idx + "_train_ID" + str(train_ID) + "_Desk_AccScor_SR" + str(SR)
    )
elif ML_model == "SVM":
    model_name = (
        ML_model
        + "_"
        + VI_idx
        + "_"
        + smooth
        + "_NoneWeight_00_Desk_AccScor_SR"
        + str(SR)
        + "_train_ID"
        + str(train_ID)
    )
elif ML_model == "RF":
    model_name = (
        ML_model
        + "_"
        + VI_idx
        + "_"
        + smooth
        + "_grid_2_Desk_AccScor_SR"
        + str(SR)
        + "_train_ID"
        + str(train_ID)
    )


trained_model = pickle.load(open(model_dir + model_name + ".sav", "rb"))

train_preds = trained_model.predict(train80_wide.iloc[:, 1:-1])
pred_colName = ML_model + "_" + VI_idx + "_" + smooth + "_preds"
A = pd.DataFrame(columns=["ID", pred_colName])
A.ID = train80_wide.ID.values
A[pred_colName] = train_preds
train_preds = A.copy()

train_preds["SR"] = SR
train_preds["train_test"] = "train"
train_preds["Vote"] = train80_wide.Vote

test_preds = trained_model.predict(test20_wide.iloc[:, 1:-1])
A = pd.DataFrame(columns=["ID", pred_colName])
A.ID = test20_wide.ID.values
A[pred_colName] = test_preds
test_preds = A.copy()
test_preds["SR"] = SR
test_preds["train_test"] = "test"
test_preds["Vote"] = test20_wide.Vote

train_test_preds = pd.concat([train_preds, test_preds])
train_test_preds.Vote = train_test_preds.Vote.astype(int)

##############################################################################

out_name = model_name + ".csv"
train_test_preds.to_csv(res_dir + out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

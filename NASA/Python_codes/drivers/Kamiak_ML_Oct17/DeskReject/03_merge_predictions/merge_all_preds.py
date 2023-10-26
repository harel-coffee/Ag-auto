import shutup  # , random

shutup.please()

import numpy as np
import pandas as pd
from datetime import date, datetime
import sys, os, os.path, shutil, h5py, time

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
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"
deskReject_dir = data_base + "DeskRejectResults/"
pred_dir = deskReject_dir + "/00_predictions/"
merged_pred_dir = deskReject_dir + "01_merged_preds/"
os.makedirs(merged_pred_dir, exist_ok=True)
#####################################################################
######
######                           Body
######
#####################################################################
pred_file_names = [x for x in os.listdir(pred_dir) if x.endswith(".csv")]
all_DL_preds = pd.DataFrame()
all_SVM_preds = pd.DataFrame()
all_KNN_preds = pd.DataFrame()
all_RF_preds = pd.DataFrame()

for a_file in sorted(pred_file_names):
    df = pd.read_csv(pred_dir + a_file)
    a_file_split = a_file.split("_")
    if a_file_split[0] in ["SVM", "KNN", "RF"]:
        for entry in a_file_split:
            if "ID" in entry:
                df["train_ID"] = entry[2]
    else:
        df["train_ID"] = a_file_split[3]
        df["SR"] = a_file_split[-1].split(".")[0]
    if a_file_split[0] == "SVM":
        all_SVM_preds = pd.concat([all_SVM_preds, df])
    elif a_file_split[0] == "KNN":
        all_KNN_preds = pd.concat([all_KNN_preds, df])
    elif a_file_split[0] == "RF":
        all_RF_preds = pd.concat([all_RF_preds, df])
    else:
        all_DL_preds = pd.concat([all_DL_preds, df])


all_DL_preds.SR = all_DL_preds.SR.astype(int)

all_preds = all_DL_preds.copy()
all_preds = pd.merge(
    all_preds,
    all_RF_preds[["ID", "RF_NDVI_SG_preds", "SR", "train_ID"]],
    on=["ID", "SR", "train_ID"],
    how="left",
)
all_preds = pd.merge(
    all_preds,
    all_KNN_preds[["ID", "KNN_NDVI_SG_preds", "SR", "train_ID"]],
    on=["ID", "SR", "train_ID"],
    how="left",
)
all_preds = pd.merge(
    all_preds,
    all_SVM_preds[["ID", "SVM_NDVI_SG_preds", "SR", "train_ID"]],
    on=["ID", "SR", "train_ID"],
    how="left",
)

out_name = "all_preds.csv"
all_preds.to_csv(merged_pred_dir + out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

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
pred_file_names = [x for x in pred_file_names if "DL" in x]
all_preds = pd.DataFrame()

for a_file in pred_file_names:
    df = pd.read_csv(pred_dir + a_file)
    a_file_split = a_file.split("_")
    df["train_ID"] = a_file_split[3]
    df["SR"] = a_file_split[-1].split(".")[0]
    all_preds = pd.concat([all_preds, df])


out_name = "all_DL_preds.csv"
all_preds.to_csv(merged_pred_dir + out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

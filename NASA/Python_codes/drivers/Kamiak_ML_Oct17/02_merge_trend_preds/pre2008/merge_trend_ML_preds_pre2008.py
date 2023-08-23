import shutup, time  # , random

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
import sys, os, os.path, shutil
import pickle, h5py

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")


####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"
in_dir = data_base + "trend_ML_preds/"

out_dir = data_base + "merged_trend_ML_preds/"
os.makedirs(out_dir, exist_ok=True)
####################################################################################
###
###      Create file names. all bacthes of the same
###      technique w/ the same smoothness will be on the same list.
###
####################################################################################
NDVI_regular_RF_files = [
    m + str(n) for m, n in zip(["NDVI_regular_RF_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_regular_RF_files = [
    m + str(n) for m, n in zip(NDVI_regular_RF_files, ["_preds_pre2008.csv"] * 40)
]

EVI_regular_RF_files = [
    m + str(n) for m, n in zip(["EVI_regular_RF_batchNumber"] * 40, list(range(1, 41)))
]
EVI_regular_RF_files = [
    m + str(n) for m, n in zip(EVI_regular_RF_files, ["_preds_pre2008.csv"] * 40)
]

NDVI_SG_RF_files = [m + str(n) for m, n in zip(["NDVI_SG_RF_batchNumber"] * 40, list(range(1, 41)))]
NDVI_SG_RF_files = [m + str(n) for m, n in zip(NDVI_SG_RF_files, ["_preds_pre2008.csv"] * 40)]

EVI_SG_RF_files = [m + str(n) for m, n in zip(["EVI_SG_RF_batchNumber"] * 40, list(range(1, 41)))]
EVI_SG_RF_files = [m + str(n) for m, n in zip(EVI_SG_RF_files, ["_preds_pre2008.csv"] * 40)]

NDVI_regular_SVM_files = [
    m + str(n) for m, n in zip(["NDVI_regular_SVM_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_regular_SVM_files = [
    m + str(n) for m, n in zip(NDVI_regular_SVM_files, ["_preds_pre2008.csv"] * 40)
]

EVI_regular_SVM_files = [
    m + str(n) for m, n in zip(["EVI_regular_SVM_batchNumber"] * 40, list(range(1, 41)))
]
EVI_regular_SVM_files = [
    m + str(n) for m, n in zip(EVI_regular_SVM_files, ["_preds_pre2008.csv"] * 40)
]

NDVI_SG_SVM_files = [
    m + str(n) for m, n in zip(["NDVI_SG_SVM_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_SG_SVM_files = [m + str(n) for m, n in zip(NDVI_SG_SVM_files, ["_preds_pre2008.csv"] * 40)]

EVI_SG_SVM_files = [m + str(n) for m, n in zip(["EVI_SG_SVM_batchNumber"] * 40, list(range(1, 41)))]
EVI_SG_SVM_files = [m + str(n) for m, n in zip(EVI_SG_SVM_files, ["_preds_pre2008.csv"] * 40)]


NDVI_regular_DL_files = [
    m + str(n) for m, n in zip(["NDVI_regular_DL_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_regular_DL_files = [
    m + str(n) for m, n in zip(NDVI_regular_DL_files, ["_preds_pre2008.csv"] * 40)
]

EVI_regular_DL_files = [
    m + str(n) for m, n in zip(["EVI_regular_DL_batchNumber"] * 40, list(range(1, 41)))
]
EVI_regular_DL_files = [
    m + str(n) for m, n in zip(EVI_regular_DL_files, ["_preds_pre2008.csv"] * 40)
]

NDVI_SG_DL_files = [m + str(n) for m, n in zip(["NDVI_SG_DL_batchNumber"] * 40, list(range(1, 41)))]
NDVI_SG_DL_files = [m + str(n) for m, n in zip(NDVI_SG_DL_files, ["_preds_pre2008.csv"] * 40)]

EVI_SG_DL_files = [m + str(n) for m, n in zip(["EVI_SG_DL_batchNumber"] * 40, list(range(1, 41)))]
EVI_SG_DL_files = [m + str(n) for m, n in zip(EVI_SG_DL_files, ["_preds_pre2008.csv"] * 40)]

NDVI_regular_KNN_files = [
    m + str(n) for m, n in zip(["NDVI_regular_KNN_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_regular_KNN_files = [
    m + str(n) for m, n in zip(NDVI_regular_KNN_files, ["_preds_pre2008.csv"] * 40)
]

EVI_regular_KNN_files = [
    m + str(n) for m, n in zip(["EVI_regular_KNN_batchNumber"] * 40, list(range(1, 41)))
]
EVI_regular_KNN_files = [
    m + str(n) for m, n in zip(EVI_regular_KNN_files, ["_preds_pre2008.csv"] * 40)
]

NDVI_SG_KNN_files = [
    m + str(n) for m, n in zip(["NDVI_SG_KNN_batchNumber"] * 40, list(range(1, 41)))
]
NDVI_SG_KNN_files = [m + str(n) for m, n in zip(NDVI_SG_KNN_files, ["_preds_pre2008.csv"] * 40)]

EVI_SG_KNN_files = [m + str(n) for m, n in zip(["EVI_SG_KNN_batchNumber"] * 40, list(range(1, 41)))]
EVI_SG_KNN_files = [m + str(n) for m, n in zip(EVI_SG_KNN_files, ["_preds_pre2008.csv"] * 40)]

### read and concatenate bacthes to form a single experiment (experiment e.g. NDVI_SG_RF)
NDVI_SG_RF, NDVI_SG_DL, NDVI_SG_KNN, NDVI_SG_SVM = (
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
)

NDVI_regular_RF, NDVI_regular_DL, NDVI_regular_KNN, NDVI_regular_SVM = (
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
)

EVI_SG_RF, EVI_SG_DL, EVI_SG_KNN, EVI_SG_SVM = (
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
)

EVI_regular_RF, EVI_regular_DL, EVI_regular_KNN, EVI_regular_SVM = (
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
    pd.DataFrame(),
)

# NDVI_SG_RF batches
for a_file in NDVI_SG_RF_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_SG_RF = pd.concat([NDVI_SG_RF, curr_df])

# EVI_SG_RF batches
for a_file in EVI_SG_RF_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_SG_RF = pd.concat([EVI_SG_RF, curr_df])

# NDVI_SG_SVM batches
for a_file in NDVI_SG_SVM_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_SG_SVM = pd.concat([NDVI_SG_SVM, curr_df])

# EVI_SG_SVM batches
for a_file in EVI_SG_SVM_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_SG_SVM = pd.concat([EVI_SG_SVM, curr_df])

# NDVI_SG_KNN batches
for a_file in NDVI_SG_KNN_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_SG_KNN = pd.concat([NDVI_SG_KNN, curr_df])

# EVI_SG_KNN batches
for a_file in EVI_SG_KNN_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_SG_KNN = pd.concat([EVI_SG_KNN, curr_df])

# NDVI_SG_DL batches
for a_file in NDVI_SG_DL_files:
    curr_df = pd.read_csv(in_dir + a_file)
    curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
    curr_df["year"] = (
        curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
    )
    curr_df.year = curr_df.year.astype(int)
    curr_df = curr_df[["ID", "year", "prob_single"]]
    curr_df.rename(columns={"prob_single": "NDVI_SG_DL_p_single"}, inplace=True)

    NDVI_SG_DL = pd.concat([NDVI_SG_DL, curr_df])

# EVI_SG_DL batches
for a_file in EVI_SG_DL_files:
    curr_df = pd.read_csv(in_dir + a_file)
    curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
    curr_df["year"] = (
        curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
    )
    curr_df.year = curr_df.year.astype(int)
    curr_df = curr_df[["ID", "year", "prob_single"]]
    curr_df.rename(columns={"prob_single": "EVI_SG_DL_p_single"}, inplace=True)
    EVI_SG_DL = pd.concat([EVI_SG_DL, curr_df])


# NDVI_regular_RF batches
for a_file in NDVI_regular_RF_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_regular_RF = pd.concat([NDVI_regular_RF, curr_df])

# EVI_regular_RF batches
for a_file in EVI_regular_RF_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_regular_RF = pd.concat([EVI_regular_RF, curr_df])

# NDVI_regular_SVM batches
for a_file in NDVI_regular_SVM_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_regular_SVM = pd.concat([NDVI_regular_SVM, curr_df])

# EVI_regular_SVM batches
for a_file in EVI_regular_SVM_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_regular_SVM = pd.concat([EVI_regular_SVM, curr_df])

# NDVI_regular_KNN batches
for a_file in NDVI_regular_KNN_files:
    curr_df = pd.read_csv(in_dir + a_file)
    NDVI_regular_KNN = pd.concat([NDVI_regular_KNN, curr_df])

# EVI_regular_KNN batches
for a_file in EVI_regular_KNN_files:
    curr_df = pd.read_csv(in_dir + a_file)
    EVI_regular_KNN = pd.concat([EVI_regular_KNN, curr_df])

# NDVI_regular_DL batches
for a_file in NDVI_regular_DL_files:
    curr_df = pd.read_csv(in_dir + a_file)
    curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
    curr_df["year"] = (
        curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
    )
    curr_df.year = curr_df.year.astype(int)
    curr_df = curr_df[["ID", "year", "prob_single"]]
    curr_df.rename(columns={"prob_single": "NDVI_regular_DL_p_single"}, inplace=True)

    NDVI_regular_DL = pd.concat([NDVI_regular_DL, curr_df])

# EVI_regular_DL batches
for a_file in EVI_regular_DL_files:
    curr_df = pd.read_csv(in_dir + a_file)
    curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
    curr_df["year"] = (
        curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
    )
    curr_df.year = curr_df.year.astype(int)
    curr_df = curr_df[["ID", "year", "prob_single"]]
    curr_df.rename(columns={"prob_single": "EVI_regular_DL_p_single"}, inplace=True)

    EVI_regular_DL = pd.concat([EVI_regular_DL, curr_df])

######
######     Merge all the MLs of a given experiment in one file.
######
NDVI_SG_RF = pd.merge(NDVI_SG_RF, NDVI_SG_SVM, on=["ID", "year"], how="left")
NDVI_SG_RF = pd.merge(NDVI_SG_RF, NDVI_SG_KNN, on=["ID", "year"], how="left")
NDVI_SG_RF = pd.merge(NDVI_SG_RF, NDVI_SG_DL, on=["ID", "year"], how="left")

EVI_SG_RF = pd.merge(EVI_SG_RF, EVI_SG_SVM, on=["ID", "year"], how="left")
EVI_SG_RF = pd.merge(EVI_SG_RF, EVI_SG_KNN, on=["ID", "year"], how="left")
EVI_SG_RF = pd.merge(EVI_SG_RF, EVI_SG_DL, on=["ID", "year"], how="left")

NDVI_regular_RF = pd.merge(NDVI_regular_RF, NDVI_regular_SVM, on=["ID", "year"], how="left")
NDVI_regular_RF = pd.merge(NDVI_regular_RF, NDVI_regular_KNN, on=["ID", "year"], how="left")
NDVI_regular_RF = pd.merge(NDVI_regular_RF, NDVI_regular_DL, on=["ID", "year"], how="left")

EVI_regular_RF = pd.merge(EVI_regular_RF, EVI_regular_SVM, on=["ID", "year"], how="left")
EVI_regular_RF = pd.merge(EVI_regular_RF, EVI_regular_KNN, on=["ID", "year"], how="left")
EVI_regular_RF = pd.merge(EVI_regular_RF, EVI_regular_DL, on=["ID", "year"], how="left")


######  Export Output
out_name = out_dir + "NDVI_SG_preds_intersect_pre2008.csv"
NDVI_SG_RF.to_csv(out_name, index=False)

out_name = out_dir + "EVI_SG_preds_intersect_pre2008.csv"
EVI_SG_RF.to_csv(out_name, index=False)

out_name = out_dir + "NDVI_regular_preds_intersect_pre2008.csv"
NDVI_regular_RF.to_csv(out_name, index=False)

out_name = out_dir + "EVI_regular_preds_intersect_pre2008.csv"
EVI_regular_RF.to_csv(out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

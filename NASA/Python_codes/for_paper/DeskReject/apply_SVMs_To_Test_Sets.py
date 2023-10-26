# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from datetime import date, datetime
from random import seed, random
import sys, os, os.path, shutil, h5py, time, pickle


# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/DeskReject/"
CSV_dir_base = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"

# %%
VI_idx = "NDVI"
ML_model = "SVM"
smooth = "SG"

# %%
train_ID = 1
SR = 3
model_name = ML_model + "_" + VI_idx + "_" + smooth + "_NoneWeight_00_Desk_AccScor_SR" + \
             str(SR) + "_train_ID" + str(train_ID)

SVM_13 = pickle.load(open(model_dir + model_name + ".sav", "rb"))

# %%
train_ID = 2
SR = 3
model_name = ML_model + "_" + VI_idx + "_" + smooth + "_NoneWeight_00_Desk_AccScor_SR" + \
             str(SR) + "_train_ID" + str(train_ID)
SVM_23 = pickle.load(open(model_dir + model_name + ".sav", "rb"))

# %%
SVM_13 == SVM_23

# %%
test_13 = pd.read_csv(CSV_dir_base + "train_test_DL_1/" + "NDVI_SG_wide_test20_split_2Bconsistent_Oct17.csv")
test_23 = pd.read_csv(CSV_dir_base + "train_test_DL_2/" + "NDVI_SG_wide_test20_split_2Bconsistent_Oct17.csv")
# test_33 = pd.read_csv(CSV_dir_base + "train_test_DL_3/" + "NDVI_SG_wide_test20_split_2Bconsistent_Oct17.csv")
# test_43 = pd.read_csv(CSV_dir_base + "train_test_DL_4/" + "NDVI_SG_wide_test20_split_2Bconsistent_Oct17.csv")
# test_53 = pd.read_csv(CSV_dir_base + "train_test_DL_5/" + "NDVI_SG_wide_test20_split_2Bconsistent_Oct17.csv")

# %%
test_13.head(2)

# %%
test_23.head(2)

# %%
sorted(test_13.ID)[:5]

# %%
sorted(test_23.ID)[:5]

# %%
print (f"{test_13.shape = }")
print (f"{test_23.shape = }")

# %%
test_13_preds = SVM_13.predict(test_13.iloc[:, 1:-1])
test_23_preds = SVM_23.predict(test_23.iloc[:, 1:-1])

# %%
test_13_res = test_13[["ID", "Vote"]].copy()
test_23_res = test_23[["ID", "Vote"]].copy()

# %%
test_13_res["SVM"] = test_13_preds
test_23_res["SVM"] = test_23_preds

# %%
test_13_res.head(2)

# %%
test_23_res.head(2)

# %%
test_23_res[test_23_res.Vote==2]

# %%
test_13_res[test_13_res.Vote==2]

# %%
test_13.shape

# %%
test_13[test_13.Vote==1].shape

# %%
test_13[test_13.Vote==2].shape

# %%

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
import shutup
shutup.please();

import pandas as pd
import numpy as np
import os, sys
import datetime
from datetime import date, datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import SGD


import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import scipy, scipy.signal
import pickle, h5py

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc


# %%
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


# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
VI_TS_dir = dir_base + "VI_TS/09_WSU_students2022Survey/"
model_dir = dir_base + "ML_Models_Oct17/"

pred_dir = VI_TS_dir + "predictions/"
os.makedirs(pred_dir, exist_ok=True)

# %%
winnerModels = pd.read_csv(dir_base + "winnerModels.csv")
winnerModels.dropna(inplace=True)

# %%
VI_idx, smooth = "NDVI", "SG"

TS_f_name = VI_idx + "_" + smooth + "_WSUStudentSurvey2022_wide_JFD.csv"
TS_df = pd.read_csv(VI_TS_dir + TS_f_name)

fig, ax = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

ax.grid(which='major', axis='y', linestyle='--')

a_TS = TS_df[TS_df.ID == "00248fe3-373c-4b49-9099-e674c92ce9d7"].copy()

ax.plot(a_TS.iloc[:, 2:].values[0], 
        linewidth=4, color="dodgerblue", label="smoothed");
ax.set_ylim([-0.3, 1.15]);

# %%
for df_idx in winnerModels.index:
    VI_idx = winnerModels.loc[df_idx, "VI_idx"]
    smooth = winnerModels.loc[df_idx, "smooth"]
    model =  winnerModels.loc[df_idx, "model"]
    print (f"{VI_idx=}, {smooth=}, {model=}")
    print ("================================================")
    winnerModel =  winnerModels.loc[df_idx, "output_name"] 

    if winnerModel.endswith(".sav"):
        TS_f_name = VI_idx + "_" + smooth + "_WSUStudentSurvey2022_wide_JFD.csv"
        wide_TS = pd.read_csv(VI_TS_dir + TS_f_name)
        ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
        predictions = ML_model.predict(wide_TS.iloc[:, 2:])
        pred_colName = model + "_" + VI_idx + "_" + smooth + "_preds"
        A = pd.DataFrame(columns=["ID", "year", pred_colName])
        A.ID = wide_TS.ID.values
        A.year = wide_TS.year.values
        A[pred_colName] = predictions
        predictions = A.copy()
        del A
    else:
        ML_model = load_model(model_dir + winnerModel)
        ML_model = load_model(model_dir + winnerModel)
        
        plot_dir = VI_TS_dir + "/plots/" + VI_idx + "_" + smooth + "/"

        f_name = VI_idx + "_" + smooth + "_WSUStudentSurvey2022_wide_JFD.csv"
        wide_TS = pd.read_csv(VI_TS_dir + f_name)
        p_filenames_clean = list(wide_TS.ID + "_" + wide_TS.year.astype(str) + ".jpg")
        predictions = pd.DataFrame({"filename": p_filenames_clean})
        predictions["prob_single"] = -1.0

        for idx in predictions.index:
            img = load_image(plot_dir + predictions.loc[idx, "filename"])
            predictions.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]
    pred_colName = VI_idx + "_" + smooth + "_" + model + "_preds"
    out_name = pred_dir + pred_colName + ".csv"
    predictions.to_csv(out_name, index=False)

# %% [markdown]
# # Merge predictions

# %%
MLs = ["SVM", "KNN", "RF", "DL"]

EVI_regular_files = [m + str(n) for m, n in zip(["EVI_regular_"]*4, MLs)]
EVI_regular_files = [m + str(n) for m, n in zip(EVI_regular_files, ["_preds.csv"]*4)]
EVI_SG_files = [m + str(n) for m, n in zip(["EVI_SG_"]*4, MLs)]
EVI_SG_files = [m + str(n) for m, n in zip(EVI_SG_files, ["_preds.csv"]*4)]

NDVI_regular_files = [m + str(n) for m, n in zip(["NDVI_regular_"]*4, MLs)]
NDVI_regular_files = [m + str(n) for m, n in zip(NDVI_regular_files, ["_preds.csv"]*4)]
NDVI_SG_files = [m + str(n) for m, n in zip(["NDVI_SG_"]*4, MLs)]
NDVI_SG_files = [m + str(n) for m, n in zip(NDVI_SG_files, ["_preds.csv"]*4)]

# %%
for ii, file_name in enumerate(NDVI_SG_files):
    if ii==0:
        NDVI_SG = pd.read_csv(pred_dir + file_name)
    else:
        curr_df = pd.read_csv(pred_dir + file_name)
        if "DL" in file_name:
            curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
            curr_df["year"] = (
                curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
            )
            curr_df.year = curr_df.year.astype(int)
            curr_df = curr_df[["ID", "year", "prob_single"]]
            curr_df.rename(columns={"prob_single": "NDVI_SG_DL_p_single"}, inplace=True)
        NDVI_SG = pd.merge(NDVI_SG, curr_df, on=["ID", "year"], how="left")
        
for ii, file_name in enumerate(EVI_SG_files):
    if ii==0:
        EVI_SG = pd.read_csv(pred_dir + file_name)
    else:
        curr_df = pd.read_csv(pred_dir + file_name)
        if "DL" in file_name:
            curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
            curr_df["year"] = (
                curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
            )
            curr_df.year = curr_df.year.astype(int)
            curr_df = curr_df[["ID", "year", "prob_single"]]
            curr_df.rename(columns={"prob_single": "EVI_SG_DL_p_single"}, inplace=True)
        EVI_SG = pd.merge(EVI_SG, curr_df, on=["ID", "year"], how="left")

# %%
for ii, file_name in enumerate(NDVI_regular_files):
    if ii==0:
        NDVI_regular = pd.read_csv(pred_dir + file_name)
    else:
        curr_df = pd.read_csv(pred_dir + file_name)
        if "DL" in file_name:
            curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
            curr_df["year"] = (
                curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
            )
            curr_df.year = curr_df.year.astype(int)
            curr_df = curr_df[["ID", "year", "prob_single"]]
            curr_df.rename(columns={"prob_single": "NDVI_regular_DL_p_single"}, inplace=True)
        NDVI_regular = pd.merge(NDVI_regular, curr_df, on=["ID", "year"], how="left")
        
for ii, file_name in enumerate(EVI_regular_files):
    if ii==0:
        EVI_regular = pd.read_csv(pred_dir + file_name)
    else:
        curr_df = pd.read_csv(pred_dir + file_name)
        if "DL" in file_name:
            curr_df["ID"] = curr_df["filename"].str.split("_", expand=True)[0]
            curr_df["year"] = (
                curr_df["filename"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
            )
            curr_df.year = curr_df.year.astype(int)
            curr_df = curr_df[["ID", "year", "prob_single"]]
            curr_df.rename(columns={"prob_single": "EVI_regular_DL_p_single"}, inplace=True)
        EVI_regular = pd.merge(EVI_regular, curr_df, on=["ID", "year"], how="left")

# %%
out_name = pred_dir + "WSUStudents2022Survey_NDVI_SG_preds.csv"
NDVI_SG.to_csv(out_name, index=False)

out_name = pred_dir + "WSUStudents2022Survey_EVI_SG_preds.csv"
EVI_SG.to_csv(out_name, index=False)

out_name = pred_dir + "WSUStudents2022Survey_NDVI_regular_preds.csv"
NDVI_regular.to_csv(out_name, index=False)

out_name = pred_dir + "WSUStudents2022Survey_EVI_regular_preds.csv"
EVI_regular.to_csv(out_name, index=False)

# %% [markdown]
# # Ground Truth

# %%
GT_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/2022_survey/WSU_Students_2022_GEE/"
GT = pd.read_csv(GT_dir + "WSU_Students_2022_GEE_data.csv")
GT.rename(columns={"GlobalID": "ID"}, inplace=True)
GT_large = GT[GT.Acres>10].copy()
GT.head(2)

# %%
sorted(list(GT.columns))

# %%
# GT = GT[["ID"]]
(list(GT.SecondSu_1.unique()))

# %%
(list(GT.FirstSur_1.unique()))

# %%

print (f"{GT_large.shape=}")
print (f"{GT.shape=}")

# %%

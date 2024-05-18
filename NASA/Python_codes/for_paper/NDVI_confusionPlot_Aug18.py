# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook is bulit by copying the ```EVI``` version.
#
# But that does not matter because
#    - ```NDVI``` works better than ```EVI```. So, we will go with this version.
#    - In this notebook we will read predictions that are already pre-computed using best models

# %%
import time  # , shutil
import numpy as np
import pandas as pd
from datetime import date

# import random
# from random import seed, random
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# import scipy, scipy.signal

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle  # , h5py
import sys, os, os.path

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# import NASA_plot_core.py as rcp

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
# pred_dir = pred_dir_base + "02_ML_preds/"

# %%

# %%
meta_6000 = pd.read_csv(meta_dir + "evaluation_set.csv")
meta_6000_moreThan10Acr = meta_6000[meta_6000.ExctAcr > 10]

print(meta_6000.shape)
print(meta_6000_moreThan10Acr.shape)
meta_6000.head(2)

# %%
out_name = SF_data_dir + "all_SF_data_concatenated.csv"
all_SF_data = pd.read_csv(SF_data_dir + "all_SF_data_concatenated.csv")
all_SF_data.head(2)

# %% [markdown]
# # Read groundTruth Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
GT_labels = pd.read_csv(training_set_dir + "groundTruth_labels_Oct17_2022.csv")
print("Unique Votes: ", GT_labels.Vote.unique())
print(len(GT_labels.ID.unique()))
GT_labels.head(2)

# %%
all_preds = pd.read_csv(pred_dir_base + "all_preds_overSample.csv")
print(f"{all_preds.shape = }")
all_preds.head(2)

# %% [markdown]
# ## Read test set and then figure which ones are labeled by experts

# %%
test20 = pd.read_csv(training_set_dir + "test20_split_2Bconsistent_Oct17.csv")
print(f"{test20.shape}")
test20.head(2)

# %%
# mistake_searchSpace = test20[test20.ID.isin(list(expertLabels.ID))]
mistake_searchSpace = test20.copy()
mistake_searchSpace.reset_index(drop=True, inplace=True)
mistake_searchSpace = pd.merge(mistake_searchSpace, meta_6000, on=["ID"], how="left")

print(f"{mistake_searchSpace.shape=}")
print(f"{mistake_searchSpace.ExctAcr.min()=}")
mistake_searchSpace.head(2)

# %%
mistake_space_preds = all_preds[all_preds.ID.isin(list(mistake_searchSpace.ID))]
mistake_space_preds.reset_index(drop=True, inplace=True)
print(mistake_space_preds.shape)
mistake_space_preds.head(2)

# %%
SG_preds = mistake_space_preds[
    ["ID", "SVM_NDVI_SG_preds", "KNN_NDVI_SG_preds", "DL_NDVI_SG_prob_point3", "RF_NDVI_SG_preds"]
].copy()
print(f"f{SG_preds.shape = }")
SG_preds.head(2)

# %% [markdown]
# # Read the TS Data

# %%
VI_idx = "NDVI"

SG_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = [
    "SG_Walla2015_" + VI_idx + "_JFD.csv",
    "SG_AdamBenton2016_" + VI_idx + "_JFD.csv",
    "SG_Grant2017_" + VI_idx + "_JFD.csv",
    "SG_FranklinYakima2018_" + VI_idx + "_JFD.csv",
]

SG_TS = pd.DataFrame()

for file in file_names:
    curr_file = pd.read_csv(SG_data_dir + file)
    curr_file["human_system_start_time"] = pd.to_datetime(curr_file["human_system_start_time"])

    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years) == 3 or len(all_years) == 2:
        proper_year = all_years[1]
    elif len(all_years) == 1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year == proper_year]
    SG_TS = pd.concat([SG_TS, curr_file])

SG_TS.reset_index(drop=True, inplace=True)
SG_TS.head(2)

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_raw = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir + fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_raw = pd.concat([landsat_raw, curr])

landsat_raw.head(2)

# %% [markdown]
# ### Subset the TSs to mistake search space

# %%
print(f"{SG_TS.shape=}")
SG_TS = SG_TS[SG_TS.ID.isin(list(mistake_searchSpace.ID))]
landsat_raw = landsat_raw[landsat_raw.ID.isin(list(mistake_searchSpace.ID))]

SG_TS.reset_index(drop=True, inplace=True)
landsat_raw.reset_index(drop=True, inplace=True)

landsat_raw = nc.add_human_start_time_by_system_start_time(landsat_raw)

print(f"{SG_TS.shape=}")
print(f"{landsat_raw.shape=}")

# %% [markdown]
# # Are mistakes in Common?

# %%
print(mistake_searchSpace.shape)
# print (expertLabels.shape)

# %%
mistake_searchSpace.head(2)

# %%
print(f"{SG_preds.shape=}")
SG_preds.head(2)

# %%
SG_preds = pd.merge(SG_preds, mistake_searchSpace, on=["ID"], how="left")

# %% [markdown]
# #### See where all predictions agree.

# %%
SG_preds["matching_preds"] = SG_preds.apply(
    lambda x: x.SVM_NDVI_SG_preds
    == x.KNN_NDVI_SG_preds
    == x.DL_NDVI_SG_prob_point3
    == x.RF_NDVI_SG_preds,
    axis=1,
)

SG_preds.head(2)

# %%
# regular_preds = regular_preds[regular_preds.matching_preds==True]
print(f"{SG_preds.shape=}")
SG_preds_MatchingMLs = SG_preds[SG_preds.matching_preds == True]
print(f"{SG_preds_MatchingMLs.shape=}")

# %%
sorted(list(SG_preds.columns))

# %%
SG_preds_MatchingMLs[~(SG_preds_MatchingMLs.RF_NDVI_SG_preds == SG_preds_MatchingMLs.Vote)]

# %%
SG_common_mistake_fields = SG_preds_MatchingMLs[~(SG_preds_MatchingMLs.RF_NDVI_SG_preds == SG_preds_MatchingMLs.Vote)]

SG_common_mistake_fields = SG_common_mistake_fields.ID.unique()
SG_common_mistake_fields

# %%
size = 15
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {
    "legend.fontsize": 15,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": size,
    "axes.titlesize": size * 1.2,
    "xtick.labelsize": size,  #  * 0.75
    "ytick.labelsize": size,  #  * 0.75
    "axes.titlepad": 10,
}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)


# %%
def plot_oneColumn_CropTitle(dt, raw_dt, titlee, _label="raw", idx="EVI", _color="dodgerblue"):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(15, 4),
        sharex=False,
        sharey="col",  # sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.35, "wspace": 0.05},
    )
    ax.grid(True)
    ax.plot(dt["human_system_start_time"], dt[idx], linewidth=4, color=_color, label=_label)

    ax.scatter(raw_dt["human_system_start_time"], raw_dt[idx], s=20, c="r", label="raw")

    ax.set_title(titlee)
    ax.set_ylabel(idx)  # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis="y", which="major")  # , labelsize = tick_FontSize)
    ax.tick_params(axis="x", which="major")  # , labelsize = tick_FontSize) #
    ax.legend(loc="upper right")
    plt.yticks(np.arange(0, 1.05, 0.2))
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.set_ylim(-0.1, 1.05)


# %%
plot_base = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"
plot_dir = plot_base + "Confusion_and_common_mistakes_Aug18/"
os.makedirs(plot_dir, exist_ok=True)

# %%
for anID in list(SG_common_mistake_fields):
    # curr_smooth = regular_TS[regular_TS.ID == anID]
    curr_smooth = SG_TS[SG_TS.ID == anID]

    curr_raw = landsat_raw[landsat_raw.ID == anID]
    curr_year = curr_smooth.human_system_start_time.dt.year.unique()[0]
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_year]

    curr_meta = meta_6000[meta_6000.ID == anID].copy()
    curr_vote = SG_preds[SG_preds.ID == anID].Vote.values[0]
    curr_pred = SG_preds[SG_preds.ID == anID].RF_NDVI_SG_preds.values[0]
    
    if curr_vote==1:
        str_vote = "single-cropped"
    else:
        str_vote = "double-cropped"
        
    if curr_pred==1:
        str_pred = "single-cropped"
    else:
        str_pred = "double-cropped"
    
    curr_crop = list(curr_meta.CropTyp)[0]
    if "," in curr_crop:
        str_split_list = curr_crop.split(",")
        curr_crop = " (" + str_split_list[1][1:] + " " + str_split_list[0] + ")"
    else:
        curr_crop = " (" + curr_crop + ")"
    title_ = ("actual: "
         + str_vote
         + ", prediction: "
         + str_pred + curr_crop
     )

    curr_plt = plot_oneColumn_CropTitle(
        dt=curr_smooth,
        raw_dt=curr_raw,
        idx="NDVI",
        titlee=title_,
        _label = "smoothed",
    )

    # final_plot_path = plot_dir + "NDVI_regular_expert/"
    final_plot_path = plot_dir + "NDVI_SG_commonMistake_Aug18/"
    os.makedirs(final_plot_path, exist_ok=True)
    fig_name = final_plot_path + anID + "_SG.pdf"
    plt.savefig(fname=fig_name, dpi=400, bbox_inches="tight")
    plt.close("all")

# %%
final_plot_path

# %% [markdown]
# # Do the correctly labeled ones here. Only DL

# %% [markdown]
# # Read the TS again

# %%
SG_TS = SG_TS[SG_TS.ID.isin(list(mistake_searchSpace.ID))]
# regular_TS = regular_TS[regular_TS.ID.isin(list(mistake_searchSpace.ID))]
landsat_raw = landsat_raw[landsat_raw.ID.isin(list(mistake_searchSpace.ID))]

SG_TS.reset_index(drop=True, inplace=True)
# regular_TS.reset_index(drop=True, inplace=True)
landsat_raw.reset_index(drop=True, inplace=True)

landsat_raw = nc.add_human_start_time_by_system_start_time(landsat_raw)

print(f"{SG_TS.shape=}")
# print (f"{regular_TS.shape=}")
print(f"{landsat_raw.shape=}")

# %%
print (f"{SG_preds.shape = }")
SG_preds.head(2)

# %%
sorted(list(SG_preds.columns))

# %%
plot_dir = plot_base + "Confusion_and_common_mistakes_Aug18/DL_confusion/"
os.makedirs(plot_dir, exist_ok=True)

# %%
DL_correct_preds = SG_preds[SG_preds.DL_NDVI_SG_prob_point3 == SG_preds.Vote].copy()

# %%
SG_preds.shape

# %%
DL_correct_preds.shape

# %%
DL_correct_preds_A1P1 = SG_preds[(SG_preds.DL_NDVI_SG_prob_point3==1) & (SG_preds.Vote==1)]
DL_correct_preds_A2P2 = SG_preds[(SG_preds.DL_NDVI_SG_prob_point3==2) & (SG_preds.Vote==2)]

DL_correct_preds_A1P2 = SG_preds[(SG_preds.DL_NDVI_SG_prob_point3==2) & (SG_preds.Vote==1)]
DL_correct_preds_A2P1 = SG_preds[(SG_preds.DL_NDVI_SG_prob_point3==1) & (SG_preds.Vote==2)]

# %%
print (DL_correct_preds_A1P1.shape)
print (DL_correct_preds_A2P2.shape)
print (DL_correct_preds_A1P2.shape)
print (DL_correct_preds_A2P1.shape)

# %%
DL_correct = pd.concat([DL_correct_preds_A1P1, DL_correct_preds_A2P2])

for anID in list(DL_correct.ID):
    curr_smooth = SG_TS[SG_TS.ID == anID]

    curr_raw = landsat_raw[landsat_raw.ID == anID]
    curr_year = curr_smooth.human_system_start_time.dt.year.unique()[0]
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_year]

    curr_meta = meta_6000[meta_6000.ID == anID].copy()
    curr_vote = SG_preds[SG_preds.ID == anID].Vote.values[0]
    curr_pred = SG_preds[SG_preds.ID == anID].DL_NDVI_SG_prob_point3.values[0]
    
    if curr_vote==1:
        str_vote = "single-cropped"
    else:
        str_vote = "double-cropped"
        
    if curr_pred==1:
        str_pred = "single-cropped"
    else:
        str_pred = "double-cropped"
        
    curr_crop = list(curr_meta.CropTyp)[0]
    if "," in curr_crop:
        str_split_list = curr_crop.split(",")
        curr_crop = " (" + str_split_list[1][1:] + " " + str_split_list[0] + ")"
    title_ = ("actual: "
         + str_vote
         + ", predicted: "
         + str_pred + curr_crop
     )

    curr_plt = plot_oneColumn_CropTitle(
        dt=curr_smooth,
        raw_dt=curr_raw,
        idx="NDVI",
        titlee=title_,
        _label = "smoothed"
    )

    if curr_vote==1:
        vote_part = "A1P1"
        sub_folder = vote_part + "/" + list(curr_meta.CropTyp)[0] + "/"
    else:
        vote_part = "A2P2"
        sub_folder = vote_part + "/" + list(curr_meta.CropTyp)[0] + "/"
    final_plot_path = plot_dir  + sub_folder
    os.makedirs(final_plot_path, exist_ok=True)
    fig_name = final_plot_path + vote_part + "_" + anID + "_SG.pdf"
    plt.savefig(fname=fig_name, dpi=400, bbox_inches="tight")
    plt.close("all")

# %%
DL_mistakes = pd.concat([DL_correct_preds_A1P2, DL_correct_preds_A2P1])

for anID in list(DL_mistakes.ID):
    curr_smooth = SG_TS[SG_TS.ID == anID]

    curr_raw = landsat_raw[landsat_raw.ID == anID]
    curr_year = curr_smooth.human_system_start_time.dt.year.unique()[0]
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_year]

    curr_meta = meta_6000[meta_6000.ID == anID].copy()
    curr_vote = SG_preds[SG_preds.ID == anID].Vote.values[0]
    curr_pred = SG_preds[SG_preds.ID == anID].DL_NDVI_SG_prob_point3.values[0]
    
    if curr_vote==1:
        str_vote = "single-cropped"
    else:
        str_vote = "double-cropped"
        
    if curr_pred==1:
        str_pred = "single-cropped"
    else:
        str_pred = "double-cropped"
        
    curr_crop = list(curr_meta.CropTyp)[0]
    if "," in curr_crop:
        str_split_list = curr_crop.split(",")
        curr_crop = " (" + str_split_list[1][1:] + " " + str_split_list[0] + ")"
    else:
        curr_crop = " (" + curr_crop + ")"
    title_ = ("actual: "
         + str_vote
         + ", predicted: "
         + str_pred + curr_crop
     )

    curr_plt = plot_oneColumn_CropTitle(
        dt=curr_smooth,
        raw_dt=curr_raw,
        idx="NDVI",
        titlee=title_,
        _label = "smoothed"
    )
    if curr_vote==1:
        vote_part = "A1P2"
        sub_folder = vote_part + "/" + list(curr_meta.CropTyp)[0] + "/"
    else:
        vote_part = "A2P1"
        sub_folder = vote_part + "/" + list(curr_meta.CropTyp)[0] + "/"
    final_plot_path = plot_dir  + sub_folder
    os.makedirs(final_plot_path, exist_ok=True)
    fig_name = final_plot_path + vote_part + "_" + anID + "_SG.pdf"
    plt.savefig(fname=fig_name, dpi=400, bbox_inches="tight")
    plt.close("all")

# %%

# %%
from platform import python_version

print(python_version())

# %%

np.version.version

# %%
pd.__version__

# %%
import scipy
import sklearn
print (f"{scipy.version.version =}")
print (f"{sklearn.__version__ =}")

# %%
import tensorflow
print (f"{tensorflow.__version__ =}")

# %%
import keras
print (f"{keras.__version__ =}")

# %%
import dtaidistance
print (f"{dtaidistance.__version__ =}")

# %%

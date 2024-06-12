# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### These are predictions from original model in the paper:
#
# The data in ```/Users/hn/Documents/01_research_data/NASA/Jun_11_2024``` is copied from kamiak:
#
# ```/data/project/agaid/h.noorazar/NASA/Data_Models_4_RegionalStat/02_ML_preds_oversampled```

# %%
import time# , shutil
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

import pickle #, h5py
import sys, os, os.path

# %%

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as rcp

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %% [markdown]
# # Read Training Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
ground_truth_labels = pd.merge(ground_truth_labels, meta, how="left", on="ID")
ground_truth_labels.head(2)

# %%
ground_truth_labels["train_test"] = "train"
ground_truth_labels.head(2)

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
test20 = pd.read_csv(training_set_dir + "test20_split_2Bconsistent_Oct17.csv")

# %%
ground_truth_labels.loc[ground_truth_labels.ID.isin(list(test20.ID.unique())), "train_test"] = "test"
ground_truth_labels.head(2)

# %%

# %%
# we want only corns and triticale
corn_names = [x for x in ground_truth_labels.CropTyp.unique() if "corn" in x]
triticale_names = [x for x in ground_truth_labels.CropTyp.unique() if "triticale" in x]


ground_truth_labels = ground_truth_labels[ground_truth_labels.CropTyp.isin(corn_names + triticale_names)]

ground_truth_labels.reset_index(drop=True, inplace=True)

print (ground_truth_labels.shape)
ground_truth_labels.head(2)

# %%
VI_idx="NDVI"

# %%
SG_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

SG_data_4_plot = pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(SG_data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    SG_data_4_plot=pd.concat([SG_data_4_plot, curr_file])

SG_data_4_plot = SG_data_4_plot[SG_data_4_plot["ID"].isin(list(ground_truth_labels.ID))].copy()
SG_data_4_plot.reset_index(drop=True, inplace=True)

SG_data_4_plot.head(2)

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

####################################################################
landsat_DF.reset_index(drop=True, inplace=True)

print (format(landsat_DF.shape[0], ",d"))
landsat_DF.head(2)

# %%
landsat_DF = landsat_DF[landsat_DF["ID"].isin(list(ground_truth_labels.ID))].copy()
landsat_DF.reset_index(drop=True, inplace=True)
print (format(len(landsat_DF.ID.unique()), ",d"))
print (format(landsat_DF.shape[0], ",d"))
landsat_DF.head(2)

# %%
landsat_DF = nc.add_human_start_time_by_system_start_time(landsat_DF)
landsat_DF.head(2)

# %%

# %%

# %% [markdown]
# ### Read DL labels
#
# The data in ```/Users/hn/Documents/01_research_data/NASA/Jun_11_2024``` is copied from kamiak:
#
# ```/data/project/agaid/h.noorazar/NASA/Data_Models_4_RegionalStat/02_ML_preds_oversampled```

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/Jun_11_2024/"
dir_ = dir_base + "02_ML_preds_oversampled/"
csv_files = os.listdir(dir_)
csv_files = [x for x in csv_files if x.endswith("csv")]
DL_csv_files = [x for x in csv_files if "DL_NDVI_SG" in x]
DL_csv_files

# %%
DL_predictions_ = pd.DataFrame()
for a_file in DL_csv_files:
    DL_predictions_ = pd.concat([DL_predictions_, pd.read_csv(dir_ + a_file)])

DL_predictions_ = DL_predictions_[["filename", "prob_single", "prob_point3"]].copy()
DL_predictions_["ID"] = DL_predictions_["filename"].str.split(".jpg", expand=True)[0]
DL_predictions_= DL_predictions_[["ID", "prob_single", "prob_point3"]].copy()
DL_predictions_.head(2)

DL_predictions_.head(2)

# %%

# %%
DL_predictions_ = DL_predictions_[DL_predictions_["ID"].isin(list(ground_truth_labels.ID))].copy()
DL_predictions_.reset_index(drop=True, inplace=True)
print (format(len(DL_predictions_.ID.unique()), ",d"))
print (format(DL_predictions_.shape[0], ",d"))
DL_predictions_.head(2)

# %%
ground_truth_labels.head(2)

# %%

# %%

# %%
#
# Plot
#
out_dir = dir_base + "plots_from_GT/"
os.makedirs(out_dir, exist_ok=True)

for an_ID in list(DL_predictions_.ID.unique()):
    smooth_field = SG_data_4_plot[SG_data_4_plot.ID==an_ID].copy()
    smooth_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                           sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0.2, 'wspace': .05});
    ax.grid(True);
    ax.plot(smooth_field['human_system_start_time'], smooth_field[VI_idx],
            linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
            label=f"smooth {VI_idx}");

    # # Raw data where we started from
    raw = landsat_DF[landsat_DF.ID==an_ID].copy()
    raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

    curr_year = smooth_field["human_system_start_time"].dt.year.unique()[0]
    raw["year"] = raw["human_system_start_time"].dt.year
    raw = raw[raw.year == curr_year].copy()
    ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

    crop_ = meta[meta.ID == an_ID]["CropTyp"].values[0]
    county_ = meta[meta.ID==an_ID]["county"].values[0]
    Irrigation_ = meta[meta.ID==an_ID]["Irrigtn"].values[0]

    label_ = DL_predictions_[DL_predictions_.ID==an_ID]["prob_point3"].values[0]
    label_prob = DL_predictions_[DL_predictions_.ID==an_ID]["prob_single"].values[0]
    label_prob = round(label_prob, 2)
    human_label = str(ground_truth_labels[ground_truth_labels.ID == an_ID].Vote.values[0])
    
    train_or_test = ground_truth_labels[ground_truth_labels.ID == an_ID].train_test.values[0]

    ax.set_title(crop_ + ", human: " + human_label + ", pred: " + \
                 label_ + " (prob_single= " + str(label_prob) + "), " + 
                 county_ + ", " + Irrigation_)
    ax.legend(loc="lower right");
    plt.ylim([-0.5, 1.2]);

    plot_dir = out_dir + crop_.replace(",", "").replace(" ", "_") + "_" + train_or_test + "/"
    
    if human_label=="1":
        sub_human_ = "A1_"
    else:
        sub_human_ = "A2_"
    
    if label_ == "single":
        subPred = "P1"
    else:
        subPred = "P2"
    
    plot_dir = plot_dir + sub_human_ + subPred + "/"
    os.makedirs(plot_dir, exist_ok=True)
    # file_name = plot_dir + crop_.replace(" ", "").replace(",", "_") + "_" + an_ID + ".pdf"
    file_name = plot_dir + an_ID + ".pdf"
    plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
    plt.close()


# %% [markdown]
# # Test and see if p=0.3 was the correct one used in the paper:

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

test20 = pd.read_csv(training_set_dir + "test20_split_2Bconsistent_Oct17.csv")

# %%
print (test20.shape)
test20.head(2)

# %%
DL_predictions_ = pd.DataFrame()
for a_file in DL_csv_files:
    DL_predictions_ = pd.concat([DL_predictions_, pd.read_csv(dir_ + a_file)])

DL_predictions_["ID"] = DL_predictions_["filename"].str.split(".jpg", expand=True)[0]
DL_predictions_= DL_predictions_[["ID", "prob_single", "prob_point3"]].copy()
DL_predictions_.head(2)

# %%
DL_predictions_test = DL_predictions_[DL_predictions_.ID.isin(list(test20.ID))].copy()
print (DL_predictions_test.shape)
DL_predictions_test.head(2)

# %%
DL_predictions_test = pd.merge(DL_predictions_test, test20, on="ID", how="left")
DL_predictions_test.head(2)

# %%
DL_predictions_test["preds"] = 1
DL_predictions_test.loc[DL_predictions_test.prob_point3=="double", "preds"] = 2

# %%
DL_predictions_test.head(2)

# %%
DL_predictions_test[DL_predictions_test.Vote == DL_predictions_test.preds].shape

# %%
632 - 623

# %%

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import h5py
import sys
import os, os.path
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tslearn.metrics import dtw as dtw_metric
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as rcp


# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


# %% [markdown]
# ### Read Meta

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
non_expert_vote_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/badData/"

out_name = non_expert_vote_dir + "nonExpert_votes.csv"
nonExpert_V= pd.read_csv(out_name)

out_name = non_expert_vote_dir + "limitCrops_nonExpert_votes.csv"
limitCrops_nonExpert_V = pd.read_csv(out_name)

# %%
limitCrops_nonExpert_V.head(2)

# %%
bad_columns = ['NDVI_TS_Name', 'Acres', 'LstSrvD', 'Form', 'Question']

nonExpert_V.drop(bad_columns, axis='columns', inplace=True)
limitCrops_nonExpert_V.drop(bad_columns, axis='columns', inplace=True)

# %%
limitCrops_nonExpert_V.head(2)

# %%
nonExpert_V=pd.merge(nonExpert_V, 
                     meta[["ID", "ExctAcr"]],
                     on=['ID'], how='left')


limitCrops_nonExpert_V=pd.merge(limitCrops_nonExpert_V, 
                                meta[["ID", "ExctAcr"]],
                                on=['ID'], how='left')

# %%
print (len(nonExpert_V.ID.unique()))
print (len(nonExpert_V[nonExpert_V.ExctAcr<10].ID.unique()))

# %% [markdown]
# # Toss small fields

# %%
nonExpert_V=nonExpert_V[nonExpert_V.ExctAcr>10]
limitCrops_nonExpert_V=limitCrops_nonExpert_V[limitCrops_nonExpert_V.ExctAcr>10]

# %% [markdown]
# ### Drop unknown field

# %%
nonExpert_V=nonExpert_V[nonExpert_V.CropTyp!="unknown"].copy()
limitCrops_nonExpert_V=limitCrops_nonExpert_V[limitCrops_nonExpert_V.CropTyp!="unknown"].copy()

# %% [markdown]
# # Write the non-expert set to the disk

# %%
out_name = meta_dir + "nonExpert_2605_votes.csv"
nonExpert_V.to_csv(out_name, index = False)

# %% [markdown]
# # Read SG files

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

# %%
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data=data[data.ID.isin(list(nonExpert_V.ID.unique()))]
data.reset_index(drop=True, inplace=True)

data.head(2)

# %% [markdown]
# # Sort the order of time-series and labels identically

# %%
data.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
nonExpert_V.sort_values(by=["ID"], inplace=True)

data.reset_index(drop=True, inplace=True)
nonExpert_V.reset_index(drop=True, inplace=True)

assert (len(data.ID.unique()) == len(nonExpert_V.ID.unique()))


# %% [markdown]
# # Widen

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
data_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(data.ID.unique())))
data_wide["ID"] = data.ID.unique()

for an_ID in data.ID.unique():
    curr_df = data[data.ID==an_ID]
    
    data_wide_indx = data_wide[data_wide.ID==an_ID].index
    data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%
data_wide.head(2)

# %%
nonExpert_V.head(2)

# %%
nonExpert_V_clean=nonExpert_V[["ID", "Vote", "CropTyp", "Irrigtn", "county", "ExctAcr"]].copy()
nonExpert_V_clean.replace("double", 2, inplace=True)
nonExpert_V_clean.replace("single", 1, inplace=True)

# %%
print (sorted(nonExpert_V_clean.CropTyp.unique()))

# %%
len(nonExpert_V_clean.ID.unique())

# %% [markdown]
# # SVM

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + 'SVM_classifier_balanced_SGEVI_00.sav'
SVM_classifier_balanced_00 = pickle.load(open(filename, 'rb'))

# %%
SVM_classifier_balanced_00_predictions = SVM_classifier_balanced_00.predict(data_wide.iloc[:, 1:])

# %%
#### add SVM result to vote table
nonExpert_V_clean["SVM_balanced_pred"]=SVM_classifier_balanced_00_predictions

# %%
nonExpert_V_clean.head(2)


# %% [markdown]
# # kNN
# Winner is **uniform** weight.

# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


# %%
filename = model_dir + "00_KNN_SG_EVI_DTW_prune_uniformWeight_9NNisBest.sav"
uniform_KNN = pickle.load(open(filename, 'rb'))

# %%
# %%time
KNN_DTW_test_predictions_uniform = uniform_KNN.predict(data_wide.iloc[:, 1:])

# %%
nonExpert_V_clean["knn_uniform_pred"]=KNN_DTW_test_predictions_uniform
nonExpert_V_clean.head(2)

# %%
sum(nonExpert_V_clean.SVM_balanced_pred==nonExpert_V_clean.knn_uniform_pred)

# %% [markdown]
# # RF

# %%
# %%time
filename = model_dir + 'SG_forest_grid_1.sav'
forest_grid_1_SG = pickle.load(open(filename, 'rb'))
RF_grid_1_predictions = forest_grid_1_SG.predict(data_wide.iloc[:, 1:])

# %%
nonExpert_V_clean["RF_G1_pred"]=RF_grid_1_predictions
nonExpert_V_clean.head(2)

# %%
sum(nonExpert_V_clean.SVM_balanced_pred==nonExpert_V_clean.RF_G1_pred)

# %% [markdown]
# # DL
#
# We are going with **prob_point7**.

# %%
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
DL_model = load_model(model_dir+'01_TL_SingleDoubleEVI_SG_train80.h5')

# %%

# %%
### We did this once. We can read from now on. 

# SG_nonExpert_V_EVI_plot_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_nonExpert_V_EVI/"

# nonExpert_df_filenames = os.listdir(SG_nonExpert_V_EVI_plot_dir)
# nonExpert_df = pd.DataFrame({'filename': nonExpert_df_filenames})
# nb_samples = nonExpert_df.shape[0]

# nonExpert_df["human_predict"] = nonExpert_df.filename.str.split("_", expand=True)[0]
# nonExpert_df["prob_single"]=-1.0
# print (nonExpert_df.shape)
# nonExpert_df.head(2)


# for idx in nonExpert_df.index:
#     img = load_image(SG_nonExpert_V_EVI_plot_dir + nonExpert_df.loc[idx, 'filename']);
#     nonExpert_df.loc[idx, 'prob_single'] = DL_model.predict(img, verbose=False)[0][0];

# for prob in [0.3, 0.4, 0.5, 0.6, 0.7]:
#     colName = "prob_point"+str(prob)[2:]
#     nonExpert_df.loc[nonExpert_df.prob_single<prob, colName] = 'double'
#     nonExpert_df.loc[nonExpert_df.prob_single>=prob, colName] = 'single'

# nonExpert_df.rename(columns={"filename": "ID"}, inplace=True)
# nonExpert_df.rename(columns={"human_predict": "Vote"}, inplace=True) 
# ID = [x[0] for x in nonExpert_df.ID.str.split(".")]
# nonExpert_df.ID=ID

# ID = ["_".join(x[1:]) for x in nonExpert_df.ID.str.split("_")]
# nonExpert_df.ID=ID

# nonExpert_df.replace("double", 2, inplace=True)
# nonExpert_df.replace("single", 1, inplace=True)

# nonExpert_df.head(2)

# out_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_nonExpert_2605_V_EVI_result/"
# os.makedirs(out_dir, exist_ok=True)
# out_name = out_dir + "01_TL_nonExpert_2605_predictions_" + VI_idx + "_SG.csv"
# nonExpert_df.to_csv(out_name, index = False)

# %%
out_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_nonExpert_2605_V_EVI_result/"
out_name = out_dir + "01_TL_nonExpert_2605_predictions_" + VI_idx + "_SG.csv"
nonExpert_df=pd.read_csv(out_name)
nonExpert_df.head(2)

# %%
nonExpert_V_clean=pd.merge(nonExpert_V_clean, nonExpert_df[["ID", "prob_point7"]],
                           on=['ID'], how='left')

# %%
nonExpert_V_clean.rename(columns={"prob_point7": "DL_pred"}, inplace=True)
nonExpert_V_clean.head(2)

# %%
nonExpert_V_clean.shape

# %%
SVM_A1P1=nonExpert_V_clean.copy()
SVM_A1P1=SVM_A1P1[SVM_A1P1.Vote==1]
SVM_A1P1=SVM_A1P1[SVM_A1P1.SVM_balanced_pred==1]
print ("There are [{0}] fields under SVM A1P1.".format(SVM_A1P1.shape[0]))
print ("Area is [{0:.2f}] under SVM A1P1.".format(SVM_A1P1.ExctAcr.sum()))

print ("=======================================================")
SVM_A2P2=nonExpert_V_clean.copy()
SVM_A2P2=SVM_A2P2[SVM_A2P2.Vote==2]
SVM_A2P2=SVM_A2P2[SVM_A2P2.SVM_balanced_pred==2]
print ("There are [{0}] fields under SVM A1P1.".format(SVM_A2P2.shape[0]))
print ("Area is [{0:.2f}] under SVM A1P1.".format(SVM_A2P2.ExctAcr.sum()))
print ("=======================================================")


SVM_A2P1=nonExpert_V_clean.copy()
SVM_A2P1=SVM_A2P1[SVM_A2P1.Vote==2]
SVM_A2P1=SVM_A2P1[SVM_A2P1.SVM_balanced_pred==1]
print ("There are [{0}] fields under SVM A2P1.".format(SVM_A2P1.shape[0]))
print ("Area is [{0:.2f}] under SVM A2P1.".format(SVM_A2P1.ExctAcr.sum()))

print ("=======================================================")
SVM_A1P2=nonExpert_V_clean.copy()
SVM_A1P2=SVM_A1P2[SVM_A1P2.Vote==1]
SVM_A1P2=SVM_A1P2[SVM_A1P2.SVM_balanced_pred==2]
print ("There are [{0}] fields under SVM A1P2.".format(SVM_A1P2.shape[0]))
print ("Area is [{0:.2f}] under SVM A1P2.".format(SVM_A1P2.ExctAcr.sum()))
print ("=======================================================")

# %%
RF_A1P1=nonExpert_V_clean.copy()
RF_A1P1=RF_A1P1[RF_A1P1.Vote==1]
RF_A1P1=RF_A1P1[RF_A1P1.RF_G1_pred==1]
print ("There are [{0}] fields under RF A1P1.".format(RF_A1P1.shape[0]))
print ("Area is [{0:.2f}] under RF A1P1.".format(RF_A1P1.ExctAcr.sum()))

print ("=======================================================")
RF_A2P2=nonExpert_V_clean.copy()
RF_A2P2=RF_A2P2[RF_A2P2.Vote==2]
RF_A2P2=RF_A2P2[RF_A2P2.RF_G1_pred==2]
print ("There are [{0}] fields under RF A1P1.".format(RF_A2P2.shape[0]))
print ("Area is [{0:.2f}] under RF A1P1.".format(RF_A2P2.ExctAcr.sum()))
print ("=======================================================")


RF_A2P1=nonExpert_V_clean.copy()
RF_A2P1=RF_A2P1[RF_A2P1.Vote==2]
RF_A2P1=RF_A2P1[RF_A2P1.RF_G1_pred==1]
print ("There are [{0}] fields under RF A2P1.".format(RF_A2P1.shape[0]))
print ("Area is [{0:.2f}] under RF A2P1.".format(RF_A2P1.ExctAcr.sum()))

print ("=======================================================")
RF_A1P2=nonExpert_V_clean.copy()
RF_A1P2=RF_A1P2[RF_A1P2.Vote==1]
RF_A1P2=RF_A1P2[RF_A1P2.RF_G1_pred==2]
print ("There are [{0}] fields under RF A1P2.".format(RF_A1P2.shape[0]))
print ("Area is [{0:.2f}] under RF A1P2.".format(RF_A1P2.ExctAcr.sum()))
print ("=======================================================")

# %%
knn_A1P1=nonExpert_V_clean.copy()
knn_A1P1=knn_A1P1[knn_A1P1.Vote==1]
knn_A1P1=knn_A1P1[knn_A1P1.knn_uniform_pred==1]
print ("There are [{0}] fields under knn A1P1.".format(knn_A1P1.shape[0]))
print ("Area is [{0:.2f}] under knn A1P1.".format(knn_A1P1.ExctAcr.sum()))

print ("=======================================================")
knn_A2P2=nonExpert_V_clean.copy()
knn_A2P2=knn_A2P2[knn_A2P2.Vote==2]
knn_A2P2=knn_A2P2[knn_A2P2.knn_uniform_pred==2]
print ("There are [{0}] fields under knn A1P1.".format(knn_A2P2.shape[0]))
print ("Area is [{0:.2f}] under knn A1P1.".format(knn_A2P2.ExctAcr.sum()))
print ("=======================================================")


knn_A2P1=nonExpert_V_clean.copy()
knn_A2P1=knn_A2P1[knn_A2P1.Vote==2]
knn_A2P1=knn_A2P1[knn_A2P1.knn_uniform_pred==1]
print ("There are [{0}] fields under knn A2P1.".format(knn_A2P1.shape[0]))
print ("Area is [{0:.2f}] under knn A2P1.".format(knn_A2P1.ExctAcr.sum()))

print ("=======================================================")
knn_A1P2=nonExpert_V_clean.copy()
knn_A1P2=knn_A1P2[knn_A1P2.Vote==1]
knn_A1P2=knn_A1P2[knn_A1P2.knn_uniform_pred==2]
print ("There are [{0}] fields under knn A1P2.".format(knn_A1P2.shape[0]))
print ("Area is [{0:.2f}] under knn A1P2.".format(knn_A1P2.ExctAcr.sum()))
print ("=======================================================")

# %%
DL_A1P1=nonExpert_V_clean.copy()
DL_A1P1=DL_A1P1[DL_A1P1.Vote==1]
DL_A1P1=DL_A1P1[DL_A1P1.DL_pred==1]
print ("There are [{0}] fields under DL A1P1.".format(DL_A1P1.shape[0]))
print ("Area is [{0:.2f}] under DL A1P1.".format(DL_A1P1.ExctAcr.sum()))

print ("=======================================================")
DL_A2P2=nonExpert_V_clean.copy()
DL_A2P2=DL_A2P2[DL_A2P2.Vote==2]
DL_A2P2=DL_A2P2[DL_A2P2.DL_pred==2]
print ("There are [{0}] fields under DL A1P1.".format(DL_A2P2.shape[0]))
print ("Area is [{0:.2f}] under DL A1P1.".format(DL_A2P2.ExctAcr.sum()))
print ("=======================================================")


DL_A2P1=nonExpert_V_clean.copy()
DL_A2P1=DL_A2P1[DL_A2P1.Vote==2]
DL_A2P1=DL_A2P1[DL_A2P1.DL_pred==1]
print ("There are [{0}] fields under DL A2P1.".format(DL_A2P1.shape[0]))
print ("Area is [{0:.2f}] under DL A2P1.".format(DL_A2P1.ExctAcr.sum()))

print ("=======================================================")
DL_A1P2=nonExpert_V_clean.copy()
DL_A1P2=DL_A1P2[DL_A1P2.Vote==1]
DL_A1P2=DL_A1P2[DL_A1P2.DL_pred==2]
print ("There are [{0}] fields under DL A1P2.".format(DL_A1P2.shape[0]))
print ("Area is [{0:.2f}] under DL A1P2.".format(DL_A1P2.ExctAcr.sum()))
print ("=======================================================")

# %%
nonExpert_V_clean.ExctAcr.sum()

# %%
(3344/111376)*100

# %%

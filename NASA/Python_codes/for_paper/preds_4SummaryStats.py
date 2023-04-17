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
import scipy, scipy.signal

from datetime import date
import time

import random
from random import seed, random
# import shutil

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle #, h5py
import sys, os, os.path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
from tslearn.metrics import dtw as dtw_metric
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %%

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
pred_dir = pred_dir_base + "02_ML_preds/"

# %% [markdown]
# # Read Fields Metadata

# %%
meta_6000 = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_6000_moreThan10Acr=meta_6000[meta_6000.ExctAcr>10]

print (meta_6000.shape)
print (meta_6000_moreThan10Acr.shape)
meta_6000.head(2)

# %%
f_names=["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]

SF_data=pd.DataFrame()

for file in f_names:
    curr_file=pd.read_csv(SF_data_dir + file)
    SF_data=pd.concat([SF_data, curr_file])
    
SF_data.head(2)

# %%
out_name = SF_data_dir + "all_SF_data_concatenated.csv"
SF_data.to_csv(out_name, index = False)
SF_data.county.unique()

# %%
print (f"{len(SF_data.ID.unique())= }")
print (f"{len(nc.filter_out_nonIrrigated(SF_data).ID.unique())= }")

# %% [markdown]
# # Read predictions

# %%
p_filenames = os.listdir(pred_dir)
p_filenames_clean = []

for a_file in p_filenames:
    if a_file.endswith(".csv"):
        p_filenames_clean += [a_file]
len(p_filenames_clean)

# %%
NDVI_regular = [x for x in p_filenames_clean if "NDVI_regular" in x]
EVI_regular  = [x for x in p_filenames_clean if "EVI_regular" in x]
NDVI_SG      = [x for x in p_filenames_clean if "NDVI_SG" in x]
EVI_SG       = [x for x in p_filenames_clean if "EVI_SG" in x]

# %%
len(NDVI_regular)==len(EVI_regular)==len(NDVI_SG)==len(EVI_SG)

# %%
SVM, KNN, DL, RF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

NDVI_SG_DL_prob = "prob_point9"
for a_file in NDVI_SG:
    a_file_DF = pd.read_csv(pred_dir+a_file)
    if a_file.split("_")[0]=="SVM":
        SVM = pd.concat([SVM, a_file_DF])
    elif a_file.split("_")[0]=="KNN":
        KNN = pd.concat([KNN, a_file_DF])
    elif a_file.split("_")[0]=="RF":
        RF = pd.concat([RF, a_file_DF])
    elif a_file.split("_")[0]=="DL":
        a_file_DF["ID"] = a_file_DF.filename.str.split("_", expand=True)[0]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[1]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[2]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[3].str.split(".", expand=True)[0]
        DL = pd.concat([DL, a_file_DF[["ID", NDVI_SG_DL_prob]]])
        
NDVI_SG_preds = pd.merge(SVM, KNN, on="ID", how='left')
NDVI_SG_preds = pd.merge(NDVI_SG_preds, DL, on="ID", how='left')
NDVI_SG_preds = pd.merge(NDVI_SG_preds, RF, on="ID", how='left')

NDVI_SG_preds.loc[NDVI_SG_preds.prob_point9 == "single", NDVI_SG_DL_prob] = 1
NDVI_SG_preds.loc[NDVI_SG_preds.prob_point9 == "double", NDVI_SG_DL_prob] = 2

new_DL_col = "DL_NDVI_SG_" + NDVI_SG_DL_prob
NDVI_SG_preds.rename(columns={NDVI_SG_DL_prob: new_DL_col}, inplace=True)

NDVI_SG_preds.head(2)

# %%
SVM, KNN, DL, RF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

EVI_SG_DL_prob = "prob_point4"
for a_file in EVI_SG:
    a_file_DF = pd.read_csv(pred_dir+a_file)
    
    if a_file.split("_")[0]=="SVM":
        SVM = pd.concat([SVM, a_file_DF])
    elif a_file.split("_")[0]=="KNN":
        KNN = pd.concat([KNN, a_file_DF])
    elif a_file.split("_")[0]=="RF":
        RF = pd.concat([RF, a_file_DF])
    elif a_file.split("_")[0]=="DL":
        a_file_DF["ID"] = a_file_DF.filename.str.split("_", expand=True)[0]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[1]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[2]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[3].str.split(".", expand=True)[0]
        DL = pd.concat([DL, a_file_DF[["ID", EVI_SG_DL_prob]]])
        
EVI_SG_preds = pd.merge(SVM, KNN, on="ID", how='left')
EVI_SG_preds = pd.merge(EVI_SG_preds, DL, on="ID", how='left')
EVI_SG_preds = pd.merge(EVI_SG_preds, RF, on="ID", how='left')

EVI_SG_preds.loc[EVI_SG_preds.prob_point4 == "single", EVI_SG_DL_prob] = 1
EVI_SG_preds.loc[EVI_SG_preds.prob_point4 == "double", EVI_SG_DL_prob] = 2

new_DL_col = "DL_EVI_SG_" + EVI_SG_DL_prob
EVI_SG_preds.rename(columns={EVI_SG_DL_prob: new_DL_col}, inplace=True)

EVI_SG_preds.head(2)

# %%
SVM, KNN, DL, RF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

EVI_regular_DL_prob = "prob_point4"
for a_file in EVI_regular:
    a_file_DF = pd.read_csv(pred_dir+a_file)
    
    if a_file.split("_")[0]=="SVM":
        SVM = pd.concat([SVM, a_file_DF])
    elif a_file.split("_")[0]=="KNN":
        KNN = pd.concat([KNN, a_file_DF])
    elif a_file.split("_")[0]=="RF":
        RF = pd.concat([RF, a_file_DF])
    elif a_file.split("_")[0]=="DL":
        a_file_DF["ID"] = a_file_DF.filename.str.split("_", expand=True)[0]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[1]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[2]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[3].str.split(".", expand=True)[0]
        DL = pd.concat([DL, a_file_DF[["ID", EVI_regular_DL_prob]]])

EVI_regular_preds = pd.merge(SVM, KNN, on="ID", how='left')
EVI_regular_preds = pd.merge(EVI_regular_preds, DL, on="ID", how='left')
EVI_regular_preds = pd.merge(EVI_regular_preds, RF, on="ID", how='left')

EVI_regular_preds.loc[EVI_regular_preds.prob_point4 == "single", EVI_regular_DL_prob] = 1
EVI_regular_preds.loc[EVI_regular_preds.prob_point4 == "double", EVI_regular_DL_prob] = 2

new_DL_col = "DL_EVI_regular_" + EVI_regular_DL_prob
EVI_regular_preds.rename(columns={EVI_regular_DL_prob: new_DL_col}, inplace=True)

EVI_regular_preds.head(2)

# %%
SVM, KNN, DL, RF = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

NDVI_regular_DL_prob = "prob_point9"
for a_file in NDVI_regular:
    a_file_DF = pd.read_csv(pred_dir + a_file)
    
    if a_file.split("_")[0] == "SVM":
        SVM = pd.concat([SVM, a_file_DF])
    elif a_file.split("_")[0] == "KNN":
        KNN = pd.concat([KNN, a_file_DF])
    elif a_file.split("_")[0] == "RF":
        RF = pd.concat([RF, a_file_DF])
    elif a_file.split("_")[0] == "DL":
        a_file_DF["ID"] = a_file_DF.filename.str.split("_", expand=True)[0]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[1]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[2]+ "_" + \
        a_file_DF.filename.str.split("_", expand=True)[3].str.split(".", expand=True)[0]
        DL = pd.concat([DL, a_file_DF[["ID", NDVI_regular_DL_prob]]])
        
NDVI_regular_preds = pd.merge(SVM, KNN, on="ID", how='left')
NDVI_regular_preds = pd.merge(NDVI_regular_preds, DL, on="ID", how='left')
NDVI_regular_preds = pd.merge(NDVI_regular_preds, RF, on="ID", how='left')

NDVI_regular_preds.loc[NDVI_regular_preds.prob_point9 == "single", NDVI_regular_DL_prob] = 1
NDVI_regular_preds.loc[NDVI_regular_preds.prob_point9 == "double", NDVI_regular_DL_prob] = 2

new_DL_col = "DL_NDVI_regular_" + NDVI_regular_DL_prob
NDVI_regular_preds.rename(columns={NDVI_regular_DL_prob: new_DL_col}, inplace=True)

NDVI_regular_preds.head(2)

# %%
all_preds = pd.merge(NDVI_regular_preds, EVI_regular_preds, on="ID", how='left')
all_preds = pd.merge(all_preds, EVI_SG_preds,  on="ID", how='left')
all_preds = pd.merge(all_preds, NDVI_SG_preds, on="ID", how='left')
print (all_preds.shape)
all_preds.head(2)

# %%
SF_data = SF_data[["ID", "CropTyp", "Acres", "ExctAcr", "Irrigtn", "LstSrvD", "DataSrc", "county"]]
SF_data = nc.filter_out_nonIrrigated(SF_data)

all_preds =          pd.merge(all_preds,          SF_data, on="ID", how='left')
EVI_SG_preds =       pd.merge(EVI_SG_preds,       SF_data, on="ID", how='left')
NDVI_SG_preds =      pd.merge(NDVI_SG_preds,      SF_data, on="ID", how='left')
EVI_regular_preds =  pd.merge(EVI_regular_preds,  SF_data, on="ID", how='left')
NDVI_regular_preds = pd.merge(NDVI_regular_preds, SF_data, on="ID", how='left')

# %%
SF_data.head(2)

# %%
(sorted(all_preds.CropTyp.unique()))

# %%
badCrops = ["christmas tree", 
            "crp/conservation",
            "developed", 
            "driving range", 
            "golf course",
            "green manure",
            "nursery, caneberry",
            "nursery, greenhouse",
            "nursery, lavender",
            "nursery, orchard/vineyard",
            "nursery, ornamental",
            "nursery, silviculture",
            "reclamation seed",
            "research station",
            "unknown"]

# %%
out_name = pred_dir_base + "all_preds.csv"
all_preds.to_csv(out_name, index = False)

out_name = pred_dir_base + "NDVI_regular_preds.csv"
NDVI_regular_preds.to_csv(out_name, index = False)

out_name = pred_dir_base + "EVI_regular_preds.csv"
EVI_regular_preds.to_csv(out_name, index = False)

out_name = pred_dir_base + "NDVI_SG_preds.csv"
NDVI_SG_preds.to_csv(out_name, index = False)

out_name = pred_dir_base + "EVI_SG_preds.csv"
EVI_SG_preds.to_csv(out_name, index = False)


out_name = SF_data_dir + "irriigated_SF_data_concatenated.csv"
SF_data.to_csv(out_name, index = False)

# %%
EVI_SG_preds.head(2)

# %%
print (EVI_SG_preds.groupby(['SVM_EVI_SG_preds'])['ExctAcr'].sum())
print ("------------------------------------------------------------------------")
print (EVI_SG_preds.groupby(['KNN_EVI_SG_preds'])['ExctAcr'].sum())
print ("------------------------------------------------------------------------")
print (EVI_SG_preds.groupby(['DL_EVI_SG_prob_point4'])['ExctAcr'].sum())
print ("------------------------------------------------------------------------")
print (EVI_SG_preds.groupby(['RF_EVI_SG_preds'])['ExctAcr'].sum())

# %%
EVI_SG_summary = pd.DataFrame(columns=list(EVI_SG_preds.columns[1:5]))
EVI_SG_summary[EVI_SG_summary.columns[0]] = EVI_SG_preds.groupby(\
                                                 [EVI_SG_summary.columns[0], "county"])['ExctAcr'].sum()

EVI_SG_summary[EVI_SG_summary.columns[1]] = EVI_SG_preds.groupby(\
             [EVI_SG_summary.columns[1], "county"])['ExctAcr'].sum()

EVI_SG_summary[EVI_SG_summary.columns[2]] = EVI_SG_preds.groupby(\
              [EVI_SG_summary.columns[2], "county"])['ExctAcr'].sum()

EVI_SG_summary[EVI_SG_summary.columns[3]] = EVI_SG_preds.groupby(\
                [EVI_SG_summary.columns[3], "county"])['ExctAcr'].sum()

#EVI_SG_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
#out_name = pred_dir_base + "EVI_SG_summary.csv"
#EVI_SG_summary.to_csv(out_name, index = False)

EVI_SG_summary.round()

# %%
NDVI_SG_summary = pd.DataFrame(columns=list(NDVI_SG_preds.columns[1:5]))
NDVI_SG_summary[NDVI_SG_summary.columns[0]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary.columns[0], "county"])['ExctAcr'].sum()

NDVI_SG_summary[NDVI_SG_summary.columns[1]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary.columns[1], "county"])['ExctAcr'].sum()

NDVI_SG_summary[NDVI_SG_summary.columns[2]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary.columns[2], "county"])['ExctAcr'].sum()

NDVI_SG_summary[NDVI_SG_summary.columns[3]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary.columns[3], "county"])['ExctAcr'].sum()


#NDVI_SG_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
# out_name = pred_dir_base + "NDVI_SG_summary.csv"
# NDVI_SG_summary.to_csv(out_name, index = False)

NDVI_SG_summary.round()

# %%
NDVI_regular_summary = pd.DataFrame(columns=list(NDVI_regular_preds.columns[1:5]))
NDVI_regular_summary[NDVI_regular_summary.columns[0]] = \
                      NDVI_regular_preds.groupby([NDVI_regular_summary.columns[0], "county"])['ExctAcr'].sum()
    
NDVI_regular_summary[NDVI_regular_summary.columns[1]] = \
                   NDVI_regular_preds.groupby([NDVI_regular_summary.columns[1], "county"])['ExctAcr'].sum()

NDVI_regular_summary[NDVI_regular_summary.columns[2]] = \
                    NDVI_regular_preds.groupby([NDVI_regular_summary.columns[2], "county"])['ExctAcr'].sum()

NDVI_regular_summary[NDVI_regular_summary.columns[3]] = \
                     NDVI_regular_preds.groupby([NDVI_regular_summary.columns[3], "county"])['ExctAcr'].sum()

# NDVI_regular_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
# out_name = pred_dir_base + "NDVI_regularular_summary.csv"
# NDVI_regular_summary.to_csv(out_name, index = False)

NDVI_regular_summary.round()

# %%
EVI_regular_summary = pd.DataFrame(columns=list(EVI_regular_preds.columns[1:5]))

EVI_regular_summary[EVI_regular_summary.columns[0]] = \
                         EVI_regular_preds.groupby([EVI_regular_summary.columns[0], "county"])['ExctAcr'].sum()

EVI_regular_summary[EVI_regular_summary.columns[1]] = \
                     EVI_regular_preds.groupby([EVI_regular_summary.columns[1], "county"])['ExctAcr'].sum()
    
EVI_regular_summary[EVI_regular_summary.columns[2]] = \
                      EVI_regular_preds.groupby([EVI_regular_summary.columns[2], "county"])['ExctAcr'].sum()
    
EVI_regular_summary[EVI_regular_summary.columns[3]] = \
                      EVI_regular_preds.groupby([EVI_regular_summary.columns[3], "county"])['ExctAcr'].sum()
# EVI_regular_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
EVI_regular_summary.round()

# %% [markdown]
# # Subset to Large Fields

# %%
EVI_SG_preds_L       = EVI_SG_preds[EVI_SG_preds.Acres>10].copy()
NDVI_SG_preds_L      = NDVI_SG_preds[NDVI_SG_preds.Acres>10].copy()
EVI_regular_preds_L  = EVI_regular_preds[EVI_regular_preds.Acres>10].copy()
NDVI_regular_preds_L = NDVI_regular_preds[NDVI_regular_preds.Acres>10].copy()

# %%
EVI_regular_summary_L = pd.DataFrame(columns=list(EVI_regular_preds_L.columns[1:5]))

EVI_regular_summary_L[EVI_regular_summary_L.columns[0]] = \
                         EVI_regular_preds_L.groupby([EVI_regular_summary_L.columns[0], "county"])['ExctAcr'].sum()

EVI_regular_summary_L[EVI_regular_summary_L.columns[1]] = \
                     EVI_regular_preds_L.groupby([EVI_regular_summary_L.columns[1], "county"])['ExctAcr'].sum()
    
EVI_regular_summary_L[EVI_regular_summary_L.columns[2]] = \
                      EVI_regular_preds_L.groupby([EVI_regular_summary_L.columns[2], "county"])['ExctAcr'].sum()
    
EVI_regular_summary_L[EVI_regular_summary_L.columns[3]] = \
                      EVI_regular_preds_L.groupby([EVI_regular_summary_L.columns[3], "county"])['ExctAcr'].sum()
# EVI_regular_summary_L.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "EVI_regular_summary_L.csv"
# EVI_regular_summary_L.to_csv(out_name, index = False)

EVI_regular_summary_L.round()

# %%
EVI_regular_summary_L_count = pd.DataFrame(columns=list(EVI_regular_preds_L.columns[1:5]))

EVI_regular_summary_L_count[EVI_regular_summary_L_count.columns[0]] = \
             EVI_regular_preds_L.groupby([EVI_regular_summary_L_count.columns[0], "county"])['ID'].count()

EVI_regular_summary_L_count[EVI_regular_summary_L_count.columns[1]] = \
             EVI_regular_preds_L.groupby([EVI_regular_summary_L_count.columns[1], "county"])['ID'].count()
    
EVI_regular_summary_L_count[EVI_regular_summary_L_count.columns[2]] = \
             EVI_regular_preds_L.groupby([EVI_regular_summary_L_count.columns[2], "county"])['ID'].count()
    
EVI_regular_summary_L_count[EVI_regular_summary_L_count.columns[3]] = \
             EVI_regular_preds_L.groupby([EVI_regular_summary_L_count.columns[3], "county"])['ID'].count()
# EVI_regular_summary_L_count.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "EVI_regular_summary_L_count.csv"
# EVI_regular_summary_L_count.to_csv(out_name, index = False)

EVI_regular_summary_L_count.round()

# %%

# %%
NDVI_regular_summary_L = pd.DataFrame(columns=list(NDVI_regular_preds_L.columns[1:5]))

NDVI_regular_summary_L[NDVI_regular_summary_L.columns[0]] = \
                         NDVI_regular_preds_L.groupby([NDVI_regular_summary_L.columns[0], "county"])['ExctAcr'].sum()

NDVI_regular_summary_L[NDVI_regular_summary_L.columns[1]] = \
                     NDVI_regular_preds_L.groupby([NDVI_regular_summary_L.columns[1], "county"])['ExctAcr'].sum()
    
NDVI_regular_summary_L[NDVI_regular_summary_L.columns[2]] = \
                      NDVI_regular_preds_L.groupby([NDVI_regular_summary_L.columns[2], "county"])['ExctAcr'].sum()
    
NDVI_regular_summary_L[NDVI_regular_summary_L.columns[3]] = \
                      NDVI_regular_preds_L.groupby([NDVI_regular_summary_L.columns[3], "county"])['ExctAcr'].sum()
# NDVI_regular_summary_L.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
NDVI_regular_summary_L.round()

# %%
NDVI_regular_summary_L_count = pd.DataFrame(columns=list(NDVI_regular_preds_L.columns[1:5]))

NDVI_regular_summary_L_count[NDVI_regular_summary_L_count.columns[0]] = \
             NDVI_regular_preds_L.groupby([NDVI_regular_summary_L_count.columns[0], "county"])['ID'].count()

NDVI_regular_summary_L_count[NDVI_regular_summary_L_count.columns[1]] = \
             NDVI_regular_preds_L.groupby([NDVI_regular_summary_L_count.columns[1], "county"])['ID'].count()
    
NDVI_regular_summary_L_count[NDVI_regular_summary_L_count.columns[2]] = \
             NDVI_regular_preds_L.groupby([NDVI_regular_summary_L_count.columns[2], "county"])['ID'].count()
    
NDVI_regular_summary_L_count[NDVI_regular_summary_L_count.columns[3]] = \
             NDVI_regular_preds_L.groupby([NDVI_regular_summary_L_count.columns[3], "county"])['ID'].count()
# NDVI_regular_summary_L_count.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "NDVI_regular_summary_L_count.csv"
# NDVI_regular_summary_L_count.to_csv(out_name, index = False)

NDVI_regular_summary_L_count.round()

# %%

# %%
EVI_SG_summary_L = pd.DataFrame(columns=list(EVI_SG_preds_L.columns[1:5]))

EVI_SG_summary_L[EVI_SG_summary_L.columns[0]] = \
                         EVI_SG_preds_L.groupby([EVI_SG_summary_L.columns[0], "county"])['ExctAcr'].sum()

EVI_SG_summary_L[EVI_SG_summary_L.columns[1]] = \
                     EVI_SG_preds_L.groupby([EVI_SG_summary_L.columns[1], "county"])['ExctAcr'].sum()
    
EVI_SG_summary_L[EVI_SG_summary_L.columns[2]] = \
                      EVI_SG_preds_L.groupby([EVI_SG_summary_L.columns[2], "county"])['ExctAcr'].sum()
    
EVI_SG_summary_L[EVI_SG_summary_L.columns[3]] = \
                      EVI_SG_preds_L.groupby([EVI_SG_summary_L.columns[3], "county"])['ExctAcr'].sum()
# EVI_SG_summary_L.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
EVI_SG_summary_L.round()

# %%
EVI_SG_summary_L_count = pd.DataFrame(columns=list(EVI_SG_preds_L.columns[1:5]))

EVI_SG_summary_L_count[EVI_SG_summary_L_count.columns[0]] = \
             EVI_SG_preds_L.groupby([EVI_SG_summary_L_count.columns[0], "county"])['ID'].count()

EVI_SG_summary_L_count[EVI_SG_summary_L_count.columns[1]] = \
             EVI_SG_preds_L.groupby([EVI_SG_summary_L_count.columns[1], "county"])['ID'].count()
    
EVI_SG_summary_L_count[EVI_SG_summary_L_count.columns[2]] = \
             EVI_SG_preds_L.groupby([EVI_SG_summary_L_count.columns[2], "county"])['ID'].count()
    
EVI_SG_summary_L_count[EVI_SG_summary_L_count.columns[3]] = \
             EVI_SG_preds_L.groupby([EVI_SG_summary_L_count.columns[3], "county"])['ID'].count()
# EVI_SG_summary_L_count.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "EVI_SG_summary_L_count.csv"
# EVI_SG_summary_L_count.to_csv(out_name, index = False)

EVI_SG_summary_L_count.round()

# %%

# %%
NDVI_SG_summary_L = pd.DataFrame(columns=list(NDVI_SG_preds_L.columns[1:5]))

NDVI_SG_summary_L[NDVI_SG_summary_L.columns[0]] = \
                         NDVI_SG_preds_L.groupby([NDVI_SG_summary_L.columns[0], "county"])['ExctAcr'].sum()


NDVI_SG_summary_L[NDVI_SG_summary_L.columns[1]] = \
                     NDVI_SG_preds_L.groupby([NDVI_SG_summary_L.columns[1], "county"])['ExctAcr'].sum()
    
NDVI_SG_summary_L[NDVI_SG_summary_L.columns[2]] = \
                      NDVI_SG_preds_L.groupby([NDVI_SG_summary_L.columns[2], "county"])['ExctAcr'].sum()
    
NDVI_SG_summary_L[NDVI_SG_summary_L.columns[3]] = \
                      NDVI_SG_preds_L.groupby([NDVI_SG_summary_L.columns[3], "county"])['ExctAcr'].sum()
# NDVI_SG_summary_L.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
NDVI_SG_summary_L.round()

# %%
NDVI_SG_summary_L_count = pd.DataFrame(columns=list(NDVI_SG_preds_L.columns[1:5]))

NDVI_SG_summary_L_count[NDVI_SG_summary_L_count.columns[0]] = \
             NDVI_SG_preds_L.groupby([NDVI_SG_summary_L_count.columns[0], "county"])['ID'].count()

NDVI_SG_summary_L_count[NDVI_SG_summary_L_count.columns[1]] = \
             NDVI_SG_preds_L.groupby([NDVI_SG_summary_L_count.columns[1], "county"])['ID'].count()
    
NDVI_SG_summary_L_count[NDVI_SG_summary_L_count.columns[2]] = \
             NDVI_SG_preds_L.groupby([NDVI_SG_summary_L_count.columns[2], "county"])['ID'].count()
    
NDVI_SG_summary_L_count[NDVI_SG_summary_L_count.columns[3]] = \
             NDVI_SG_preds_L.groupby([NDVI_SG_summary_L_count.columns[3], "county"])['ID'].count()
# NDVI_SG_summary_L_count.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "NDVI_SG_summary_L_count.csv"
# NDVI_SG_summary_L_count.to_csv(out_name, index = False)

NDVI_SG_summary_L_count.round()

# %%

# %%
SF_data.reset_index(inplace=True, drop=True)
SF_data.head(2)

# %%
SF_data_L = SF_data[SF_data.Acres>10].copy()
SF_data_S = SF_data[SF_data.Acres<=10].copy()

# %%
print (f"{SF_data_L.ExctAcr.sum()=}")
print (f"{SF_data_S.ExctAcr.sum()=}")


# %% [markdown]
# ## Crop-Wise Stats:
#
# DFs so far we have to use here are
#    - ```all_preds```
#    - ```NDVI_regular_preds```
#    - ```EVI_regular_preds```
#    - ```NDVI_SG_preds```
#    - ```EVI_SG_preds```
#    
#    - ```NDVI_regular_preds_L```
#    - ```EVI_regular_preds_L```
#    - ```NDVI_SG_preds_L```
#    - ```EVI_SG_preds_L```

# %%
def group_sum_area(df, group_cols):
    """ groups by two columns given by group_cols.
    The second column in group_cols must be "CropTyp"
    and the first one is something like
                  SVM_NDVI_SG_preds
                  SVM_NDVI_regular_preds
                  SVM_EVI_SG_preds
                  SVM_EVI_regular_preds
    """
    col = df.groupby([group_cols[0], group_cols[1]])['ExctAcr'].sum().reset_index(
                            name=group_cols[0]+'_acr_sum')
    col.rename(columns={group_cols[0]: "label",
                        group_cols[0]+'_acr_sum':group_cols[0]}, inplace=True)
    return (col)


# %%
NDVI_SG_crop_summary_L = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[4], "CropTyp"])

NDVI_SG_crop_summary_L = pd.concat([NDVI_SG_crop_summary_L, col1])
NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col2, on=(["label", "CropTyp"]), how='left')
NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col3, on=(["label", "CropTyp"]), how='left')
NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col4, on=(["label", "CropTyp"]), how='left')

NDVI_SG_crop_summary_L.head(2)

# %%
EVI_SG_crop_summary_L = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_SG_preds_L, [EVI_SG_preds_L.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_SG_preds_L, [EVI_SG_preds_L.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_SG_preds_L, [EVI_SG_preds_L.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_SG_preds_L, [EVI_SG_preds_L.columns[4], "CropTyp"])

EVI_SG_crop_summary_L = pd.concat([EVI_SG_crop_summary_L, col1])
EVI_SG_crop_summary_L = pd.merge(EVI_SG_crop_summary_L, col2, on=(["label", "CropTyp"]), how='left')
EVI_SG_crop_summary_L = pd.merge(EVI_SG_crop_summary_L, col3, on=(["label", "CropTyp"]), how='left')
EVI_SG_crop_summary_L = pd.merge(EVI_SG_crop_summary_L, col4, on=(["label", "CropTyp"]), how='left')

EVI_SG_crop_summary_L.head(2)

# %%
EVI_regular_crop_summary_L = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_regular_preds_L, [EVI_regular_preds_L.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_regular_preds_L, [EVI_regular_preds_L.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_regular_preds_L, [EVI_regular_preds_L.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_regular_preds_L, [EVI_regular_preds_L.columns[4], "CropTyp"])

EVI_regular_crop_summary_L = pd.concat([EVI_regular_crop_summary_L, col1])
EVI_regular_crop_summary_L = pd.merge(EVI_regular_crop_summary_L, col2, on=(["label", "CropTyp"]), how='left')
EVI_regular_crop_summary_L = pd.merge(EVI_regular_crop_summary_L, col3, on=(["label", "CropTyp"]), how='left')
EVI_regular_crop_summary_L = pd.merge(EVI_regular_crop_summary_L, col4, on=(["label", "CropTyp"]), how='left')

EVI_regular_crop_summary_L.head(2)

# %%
NDVI_regular_crop_summary_L = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_regular_preds_L, [NDVI_regular_preds_L.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_regular_preds_L, [NDVI_regular_preds_L.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_regular_preds_L, [NDVI_regular_preds_L.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_regular_preds_L, [NDVI_regular_preds_L.columns[4], "CropTyp"])

NDVI_regular_crop_summary_L = pd.concat([NDVI_regular_crop_summary_L, col1])
NDVI_regular_crop_summary_L = pd.merge(NDVI_regular_crop_summary_L, col2, on=(["label", "CropTyp"]), how='left')
NDVI_regular_crop_summary_L = pd.merge(NDVI_regular_crop_summary_L, col3, on=(["label", "CropTyp"]), how='left')
NDVI_regular_crop_summary_L = pd.merge(NDVI_regular_crop_summary_L, col4, on=(["label", "CropTyp"]), how='left')

NDVI_regular_crop_summary_L.head(2)

# %% [markdown]
# ### EVI Regular SVM

# %%
EVI_regular_crop_summary_L.sort_values(by=["label", "SVM_EVI_regular_preds"], ascending=[True, False]).round().head(5)

# %%
EVI_regular_crop_summary_L[EVI_regular_crop_summary_L.label==2].sort_values(by=["label", "SVM_EVI_regular_preds"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI Regular KNN

# %%
EVI_regular_crop_summary_L.sort_values(by=["label", "KNN_EVI_regular_preds"], ascending=[True, False]).round().head(5)

# %%
EVI_regular_crop_summary_L[EVI_regular_crop_summary_L.label==2].sort_values(by=["label", "KNN_EVI_regular_preds"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI Regular DL

# %%
EVI_regular_crop_summary_L.sort_values(by=["label", "DL_EVI_regular_prob_point4"], 
                                       ascending=[True, False]).round().head(5)

# %%
EVI_regular_crop_summary_L[EVI_regular_crop_summary_L.label==2].sort_values(by=["label", 
                                                                                "DL_EVI_regular_prob_point4"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI Regular RF

# %%
EVI_regular_crop_summary_L.sort_values(by=["label", "RF_EVI_regular_preds"], 
                                       ascending=[True, False]).round().head(5)

# %%
EVI_regular_crop_summary_L[EVI_regular_crop_summary_L.label==2].sort_values(by=["label", 
                                                                                "RF_EVI_regular_preds"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI Regular SVM

# %%
NDVI_regular_crop_summary_L.sort_values(by=["label", "SVM_NDVI_regular_preds"], 
                                        ascending=[True, False]).round().head(5)

# %%
NDVI_regular_crop_summary_L[NDVI_regular_crop_summary_L.label==2].sort_values(by=["label", "SVM_NDVI_regular_preds"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI Regular kNN

# %%
NDVI_regular_crop_summary_L.sort_values(by=["label", "KNN_NDVI_regular_preds"], 
                                        ascending=[True, False]).round().head(5)

# %%
NDVI_regular_crop_summary_L[NDVI_regular_crop_summary_L.label==2].sort_values(by=["label", "KNN_NDVI_regular_preds"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI Regular DL

# %%
NDVI_regular_crop_summary_L.sort_values(by=["label", "DL_NDVI_regular_prob_point9"], 
                                        ascending=[True, False]).round().head(5)

# %%
NDVI_regular_crop_summary_L[NDVI_regular_crop_summary_L.label==2].sort_values(
                                                by=["label", "DL_NDVI_regular_prob_point9"], 
                                                ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI Regular RF

# %%
NDVI_regular_crop_summary_L.sort_values(by=["label", "RF_NDVI_regular_preds"], 
                                        ascending=[True, False]).round().head(5)

# %%
NDVI_regular_crop_summary_L[NDVI_regular_crop_summary_L.label==2].sort_values(
                                                by=["label", "RF_NDVI_regular_preds"], 
                                                ascending=[True, False]).round().head(5)

# %%

# %% [markdown]
# ### EVI SG SVM

# %%
EVI_SG_crop_summary_L.sort_values(by=["label", "SVM_EVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
EVI_SG_crop_summary_L[EVI_SG_crop_summary_L.label==2].sort_values(by=["label", "SVM_EVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI SG kNN

# %%
EVI_SG_crop_summary_L.sort_values(by=["label", "KNN_EVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
EVI_SG_crop_summary_L[EVI_SG_crop_summary_L.label==2].sort_values(by=["label", "KNN_EVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI SG DL

# %%
EVI_SG_crop_summary_L.sort_values(by=["label", "DL_EVI_SG_prob_point4"], ascending=[True, False]).round().head(5)

# %%
EVI_SG_crop_summary_L[EVI_SG_crop_summary_L.label==2].sort_values(by=["label", "DL_EVI_SG_prob_point4"], 
                                                                            ascending=[True, False]).round().head(5)

# %% [markdown]
# ### EVI SG RF

# %%
EVI_SG_crop_summary_L.sort_values(by=["label", "RF_EVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
EVI_SG_crop_summary_L[EVI_SG_crop_summary_L.label==2].sort_values(by=["label", "RF_EVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %%

# %% [markdown]
# ### NDVI SG SVM

# %%
NDVI_SG_crop_summary_L.sort_values(by=["label", "SVM_NDVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
NDVI_SG_crop_summary_L[NDVI_SG_crop_summary_L.label==2].sort_values(by=["label", "SVM_NDVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI SG kNN

# %%
NDVI_SG_crop_summary_L.sort_values(by=["label", "KNN_NDVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
NDVI_SG_crop_summary_L[NDVI_SG_crop_summary_L.label==2].sort_values(by=["label", "KNN_NDVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI SG DL

# %%
NDVI_SG_crop_summary_L.sort_values(by=["label", "DL_NDVI_SG_prob_point9"], ascending=[True, False]).round().head(5)

# %%
NDVI_SG_crop_summary_L[NDVI_SG_crop_summary_L.label==2].sort_values(by=["label", "DL_NDVI_SG_prob_point9"], 
                                                                  ascending=[True, False]).round().head(5)

# %% [markdown]
# ### NDVI SG RF

# %%
NDVI_SG_crop_summary_L.sort_values(by=["label", "RF_NDVI_SG_preds"], ascending=[True, False]).round().head(5)

# %%
NDVI_SG_crop_summary_L[NDVI_SG_crop_summary_L.label==2].sort_values(by=["label", "RF_NDVI_SG_preds"], 
                                                                  ascending=[True, False]).round().head(5)

# %%

# %% [markdown]
# # Crops Percentage Wise
#
#    - For this we need to filter by last survey date so that labels are correct.

# %%
# NDVI_SG_crop_summary_L = pd.DataFrame(columns=["label", "CropTyp"])

# col1 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[1], "CropTyp"])
# col2 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[2], "CropTyp"])
# col3 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[3], "CropTyp"])
# col4 = group_sum_area(NDVI_SG_preds_L, [NDVI_SG_preds_L.columns[4], "CropTyp"])

# NDVI_SG_crop_summary_L = pd.concat([NDVI_SG_crop_summary_L, col1])
# NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col2, on=(["label", "CropTyp"]), how='left')
# NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col3, on=(["label", "CropTyp"]), how='left')
# NDVI_SG_crop_summary_L = pd.merge(NDVI_SG_crop_summary_L, col4, on=(["label", "CropTyp"]), how='left')

# NDVI_SG_crop_summary_L.head(2)

# %%
NDVI_SG_crop_summary_L.head(2)

# %%
### Filter by last survey date 
county_year_dict = {"Adams":2016, 
                    "Benton":2016,
                    "Frankling":2018,
                    "Grant": 2017, 
                    "Walla Walla":2015,
                    "Yakima":2018}


NDVI_SG_preds_L_LSD = pd.DataFrame()
EVI_SG_preds_L_LSD = pd.DataFrame()
NDVI_regular_preds_L_LSD = pd.DataFrame()
EVI_regular_preds_L_LSD = pd.DataFrame()

SF_data_L_LSD = pd.DataFrame()

for a_county in county_year_dict.keys():
    curr_county_DF = NDVI_SG_preds_L[NDVI_SG_preds_L.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    NDVI_SG_preds_L_LSD = pd.concat([NDVI_SG_preds_L_LSD, curr_county_DF])
NDVI_SG_preds_L_LSD.reset_index(drop=True, inplace=True)


for a_county in county_year_dict.keys():
    curr_county_DF = EVI_SG_preds_L[EVI_SG_preds_L.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    EVI_SG_preds_L_LSD = pd.concat([EVI_SG_preds_L_LSD, curr_county_DF])
EVI_SG_preds_L_LSD.reset_index(drop=True, inplace=True)

for a_county in county_year_dict.keys():
    curr_county_DF = NDVI_regular_preds_L[NDVI_regular_preds_L.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    NDVI_regular_preds_L_LSD = pd.concat([NDVI_regular_preds_L_LSD, curr_county_DF])
NDVI_regular_preds_L_LSD.reset_index(drop=True, inplace=True)


for a_county in county_year_dict.keys():
    curr_county_DF = EVI_regular_preds_L[EVI_regular_preds_L.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    EVI_regular_preds_L_LSD = pd.concat([EVI_regular_preds_L_LSD, curr_county_DF])
EVI_regular_preds_L_LSD.reset_index(drop=True, inplace=True)

for a_county in county_year_dict.keys():
    curr_county_DF = SF_data[SF_data.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    SF_data_L_LSD = pd.concat([SF_data_L_LSD, curr_county_DF])

SF_data_L_LSD = SF_data_L_LSD[SF_data_L_LSD.Acres>10].copy()
SF_data_L_LSD.reset_index(drop=True, inplace=True)


# %%
EVI_regular_preds_L_LSD.head(2)

# %% [markdown]
# #### Compute Total area per crop 

# %%
SF_data_L_LSD_grp_area = SF_data_L_LSD.groupby(['CropTyp'])['ExctAcr'].sum()
SF_data_L_LSD_grp_area = pd.DataFrame(SF_data_L_LSD_grp_area)
SF_data_L_LSD_grp_area.reset_index(drop=False, inplace=True)

out_name = pred_dir_base + "area_per_crop_LargeFields_LSD.csv"
SF_data_L_LSD_grp_area.to_csv(out_name, index = False)

SF_data_L_LSD_grp_area.head(2)

# %%
print (len(SF_data_L_LSD.ID.unique()))
print (len(NDVI_SG_preds_L.ID.unique()))

NDVI_SG_preds_L_LDS = NDVI_SG_preds_L[NDVI_SG_preds_L.ID.isin(list(SF_data_L_LSD.ID))]

print (len(NDVI_SG_preds_L_LDS.ID.unique()))

# %%

# %%
NDVI_SG_crop_summary_L_LSD = pd.DataFrame(columns=["label", "CropTyp"])
NDVI_SG_preds_L_LDS = NDVI_SG_preds_L[NDVI_SG_preds_L.ID.isin(list(SF_data_L_LSD.ID))]


col1 = group_sum_area(NDVI_SG_preds_L_LDS, [NDVI_SG_preds_L_LDS.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_SG_preds_L_LDS, [NDVI_SG_preds_L_LDS.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_SG_preds_L_LDS, [NDVI_SG_preds_L_LDS.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_SG_preds_L_LDS, [NDVI_SG_preds_L_LDS.columns[4], "CropTyp"])

NDVI_SG_crop_summary_L_LSD = pd.concat([NDVI_SG_crop_summary_L_LSD, col1])
NDVI_SG_crop_summary_L_LSD = pd.merge(
    NDVI_SG_crop_summary_L_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_L_LSD = pd.merge(
    NDVI_SG_crop_summary_L_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_L_LSD = pd.merge(
    NDVI_SG_crop_summary_L_LSD, col4, on=(["label", "CropTyp"]), how="left"
)


NDVI_SG_crop_summary_L_LSD = pd.merge(
    NDVI_SG_crop_summary_L_LSD, SF_data_L_LSD_grp_area, on=(["CropTyp"]), how="left"
)

NDVI_SG_crop_summary_L_LSD.iloc[:,2:6] = 100* (NDVI_SG_crop_summary_L_LSD.iloc[:,2:6].div(
                                                       NDVI_SG_crop_summary_L_LSD.ExctAcr, axis=0))

NDVI_SG_crop_summary_L_LSD.head(2)

# %%
print (f"{NDVI_SG_crop_summary_L_LSD.SVM_NDVI_SG_preds.max()=}")
print (f"{NDVI_SG_crop_summary_L_LSD.SVM_NDVI_SG_preds.idxmax()=}")
print ("")
print (NDVI_SG_crop_summary_L_LSD.iloc[NDVI_SG_crop_summary_L_LSD.SVM_NDVI_SG_preds.idxmax()])

# %%
EVI_SG_crop_summary_L_LSD = pd.DataFrame(columns=["label", "CropTyp"])
EVI_SG_preds_L_LDS = EVI_SG_preds_L[EVI_SG_preds_L.ID.isin(list(SF_data_L_LSD.ID))]


col1 = group_sum_area(EVI_SG_preds_L_LDS, [EVI_SG_preds_L_LDS.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_SG_preds_L_LDS, [EVI_SG_preds_L_LDS.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_SG_preds_L_LDS, [EVI_SG_preds_L_LDS.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_SG_preds_L_LDS, [EVI_SG_preds_L_LDS.columns[4], "CropTyp"])

EVI_SG_crop_summary_L_LSD = pd.concat([EVI_SG_crop_summary_L_LSD, col1])
EVI_SG_crop_summary_L_LSD = pd.merge(
    EVI_SG_crop_summary_L_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_L_LSD = pd.merge(
    EVI_SG_crop_summary_L_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_L_LSD = pd.merge(
    EVI_SG_crop_summary_L_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_SG_crop_summary_L_LSD = pd.merge(
    EVI_SG_crop_summary_L_LSD, SF_data_L_LSD_grp_area, on=(["CropTyp"]), how="left"
)

EVI_SG_crop_summary_L_LSD.iloc[:,2:6] = 100 * (EVI_SG_crop_summary_L_LSD.iloc[:,2:6].div(
                                                       EVI_SG_crop_summary_L_LSD.ExctAcr, axis=0))

EVI_SG_crop_summary_L_LSD.head(2)

# %%
NDVI_regular_crop_summary_L_LSD = pd.DataFrame(columns=["label", "CropTyp"])
NDVI_regular_preds_L_LDS = NDVI_regular_preds_L[NDVI_regular_preds_L.ID.isin(list(SF_data_L_LSD.ID))]

col1 = group_sum_area(NDVI_regular_preds_L_LDS, [NDVI_regular_preds_L_LDS.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_regular_preds_L_LDS, [NDVI_regular_preds_L_LDS.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_regular_preds_L_LDS, [NDVI_regular_preds_L_LDS.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_regular_preds_L_LDS, [NDVI_regular_preds_L_LDS.columns[4], "CropTyp"])

NDVI_regular_crop_summary_L_LSD = pd.concat([NDVI_regular_crop_summary_L_LSD, col1])
NDVI_regular_crop_summary_L_LSD = pd.merge(
    NDVI_regular_crop_summary_L_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_L_LSD = pd.merge(
    NDVI_regular_crop_summary_L_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_L_LSD = pd.merge(
    NDVI_regular_crop_summary_L_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

NDVI_regular_crop_summary_L_LSD = pd.merge(
    NDVI_regular_crop_summary_L_LSD, SF_data_L_LSD_grp_area, on=(["CropTyp"]), how="left"
)

NDVI_regular_crop_summary_L_LSD.iloc[:,2:6] = 100 * (NDVI_regular_crop_summary_L_LSD.iloc[:,2:6].div(
                                                             NDVI_regular_crop_summary_L_LSD.ExctAcr, axis=0))

NDVI_regular_crop_summary_L_LSD.head(2)

# %%
EVI_regular_crop_summary_L_LSD = pd.DataFrame(columns=["label", "CropTyp"])
EVI_regular_preds_L_LDS = EVI_regular_preds_L[EVI_regular_preds_L.ID.isin(list(SF_data_L_LSD.ID))]

col1 = group_sum_area(EVI_regular_preds_L_LDS, [EVI_regular_preds_L_LDS.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_regular_preds_L_LDS, [EVI_regular_preds_L_LDS.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_regular_preds_L_LDS, [EVI_regular_preds_L_LDS.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_regular_preds_L_LDS, [EVI_regular_preds_L_LDS.columns[4], "CropTyp"])

EVI_regular_crop_summary_L_LSD = pd.concat([EVI_regular_crop_summary_L_LSD, col1])
EVI_regular_crop_summary_L_LSD = pd.merge(
    EVI_regular_crop_summary_L_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_regular_crop_summary_L_LSD = pd.merge(
    EVI_regular_crop_summary_L_LSD, col3, on=(["label", "CropTyp"]), how="left"
)

EVI_regular_crop_summary_L_LSD = pd.merge(
    EVI_regular_crop_summary_L_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_regular_crop_summary_L_LSD = pd.merge(
    EVI_regular_crop_summary_L_LSD, SF_data_L_LSD_grp_area, on=(["CropTyp"]), how="left"
)


EVI_regular_crop_summary_L_LSD.iloc[:,2:6] = 100 * (EVI_regular_crop_summary_L_LSD.iloc[:,2:6].div(
                                                                 EVI_regular_crop_summary_L_LSD.ExctAcr, axis=0))

EVI_regular_crop_summary_L_LSD.head(2)

# %%
NDVI_SG_crop_summary_L_LSD.head(2)

# %%
EVI_regular_crop_summary_L_LSD_2cropped = EVI_regular_crop_summary_L_LSD[
                                                EVI_regular_crop_summary_L_LSD.label==2].copy()

NDVI_regular_crop_summary_L_LSD_2cropped = NDVI_regular_crop_summary_L_LSD[
                                                NDVI_regular_crop_summary_L_LSD.label==2].copy()

EVI_SG_crop_summary_L_LSD_2cropped = EVI_SG_crop_summary_L_LSD[
                                                EVI_SG_crop_summary_L_LSD.label==2].copy()

NDVI_SG_crop_summary_L_LSD_2cropped = NDVI_SG_crop_summary_L_LSD[
                                                NDVI_SG_crop_summary_L_LSD.label==2].copy()

# %%
NDVI_SG_crop_summary_L_LSD_2cropped.reset_index(inplace=True, drop=True)
EVI_SG_crop_summary_L_LSD_2cropped.reset_index(inplace=True, drop=True)

EVI_regular_crop_summary_L_LSD_2cropped.reset_index(inplace=True, drop=True)
NDVI_regular_crop_summary_L_LSD_2cropped.reset_index(inplace=True, drop=True)

# %%
print (NDVI_SG_crop_summary_L_LSD_2cropped.shape)
print (EVI_SG_crop_summary_L_LSD_2cropped.shape)

print (NDVI_regular_crop_summary_L_LSD_2cropped.shape)
print (EVI_regular_crop_summary_L_LSD_2cropped.shape)

NDVI_SG_crop_summary_L_LSD_2cropped.head(2)

# %%
NDVI_SG_crop_summary_L_LSD_2cropped.sort_values(by=["SVM_NDVI_SG_preds"], ascending=[False]).head(10)

# %%
print (f"{NDVI_SG_crop_summary_L_LSD_2cropped.SVM_NDVI_SG_preds.max()=}")
print (f"{NDVI_SG_crop_summary_L_LSD_2cropped.SVM_NDVI_SG_preds.idxmax()=}")
print ("")
print (NDVI_SG_crop_summary_L_LSD_2cropped.iloc[NDVI_SG_crop_summary_L_LSD_2cropped.SVM_NDVI_SG_preds.idxmax()])

# %%
NDVI_SG_crop_summary_L_LSD_2cropped.head(2)

# %%

# %%
NDVI_SG_crop_summary_L_LSD_2cropped.fillna(0, inplace=True)
EVI_SG_crop_summary_L_LSD_2cropped.fillna(0, inplace=True)
NDVI_regular_crop_summary_L_LSD_2cropped.fillna(0, inplace=True)
EVI_regular_crop_summary_L_LSD_2cropped.fillna(0, inplace=True)

# %%

# %%
size = 10
title_FontSize = 5
tick_legend_FontSize = 10 
label_FontSize = 14

params = {'legend.fontsize': tick_legend_FontSize, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': tick_legend_FontSize*1.2,
          'axes.titlesize': tick_legend_FontSize*1.5,
          'xtick.labelsize': tick_legend_FontSize, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize, #  * 0.75
          'axes.titlepad': 10}

plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(NDVI_SG_crop_summary_L_LSD_2cropped.CropTyp, 
        NDVI_SG_crop_summary_L_LSD_2cropped.SVM_NDVI_SG_preds, color ='dodgerblue',
        width = 0.4)

plt.xticks(rotation = 90)
plt.xlabel("crop type")
plt.ylabel("double-cropped (%)")
plt.title("SVM (5-step NDVI) predictions")
plt.show()

# %%
badCrops = ["christmas tree", 
            "crp/conservation",
            "developed", 
            "driving range", 
            "golf course",
            "green manure",
            "nursery, caneberry",
            "nursery, greenhouse",
            "nursery, lavender",
            "nursery, orchard/vineyard",
            "nursery, ornamental",
            "nursery, silviculture",
            "research station"]

NDVI_SG_crop_summary_L_LSD_2cropped = NDVI_SG_crop_summary_L_LSD_2cropped[\
                                          ~NDVI_SG_crop_summary_L_LSD_2cropped.CropTyp.isin(badCrops)]

EVI_SG_crop_summary_L_LSD_2cropped = EVI_SG_crop_summary_L_LSD_2cropped[\
                                            ~EVI_SG_crop_summary_L_LSD_2cropped.CropTyp.isin(badCrops)]

NDVI_regular_crop_summary_L_LSD_2cropped = NDVI_regular_crop_summary_L_LSD_2cropped[\
                                            ~NDVI_regular_crop_summary_L_LSD_2cropped.CropTyp.isin(badCrops)]

EVI_regular_crop_summary_L_LSD_2cropped = EVI_regular_crop_summary_L_LSD_2cropped[\
                                            ~EVI_regular_crop_summary_L_LSD_2cropped.CropTyp.isin(badCrops)]

# %%
plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(NDVI_SG_crop_summary_L_LSD_2cropped.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(NDVI_SG_crop_summary_L_LSD_2cropped.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, NDVI_SG_crop_summary_L_LSD_2cropped.SVM_NDVI_SG_preds,
         color ='dodgerblue', width = bar_width_, label="SVM")

axs.bar(X_axis - bar_width_, NDVI_SG_crop_summary_L_LSD_2cropped.KNN_NDVI_SG_preds, 
        color ='red', width = bar_width_, label="kNN")

axs.bar(X_axis, NDVI_SG_crop_summary_L_LSD_2cropped.DL_NDVI_SG_prob_point9, 
        color ='k', width = bar_width_, label="DL")

axs.bar(X_axis + bar_width_, NDVI_SG_crop_summary_L_LSD_2cropped.RF_NDVI_SG_preds, 
        color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, NDVI_SG_crop_summary_L_LSD_2cropped.CropTyp)

axs.set_ylabel("double-cropped (%)")
axs.set_xlabel("crop type")
axs.set_title("5-step NDVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");

file_name = plot_dir + "SG_NDVI_cropWise_precent.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);


plt.show()

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(EVI_SG_crop_summary_L_LSD_2cropped.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(EVI_SG_crop_summary_L_LSD_2cropped.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, EVI_SG_crop_summary_L_LSD_2cropped.SVM_EVI_SG_preds, 
        color ='dodgerblue', width = bar_width_, label="SVM")

axs.bar(X_axis - bar_width_, EVI_SG_crop_summary_L_LSD_2cropped.KNN_EVI_SG_preds,
        color ='red', width = bar_width_, label="kNN")

axs.bar(X_axis, EVI_SG_crop_summary_L_LSD_2cropped.DL_EVI_SG_prob_point4,
         color ='k', width = bar_width_, label="DL")

axs.bar(X_axis + bar_width_, EVI_SG_crop_summary_L_LSD_2cropped.RF_EVI_SG_preds,
        color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, EVI_SG_crop_summary_L_LSD_2cropped.CropTyp)

axs.set_ylabel("double-cropped (%)")
axs.set_xlabel("crop type")
axs.set_title("5-step EVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");

file_name = plot_dir + "SG_EVI_cropWise_precent.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(EVI_regular_crop_summary_L_LSD_2cropped.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(EVI_regular_crop_summary_L_LSD_2cropped.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, EVI_regular_crop_summary_L_LSD_2cropped.SVM_EVI_regular_preds, 
        color ='dodgerblue', width = bar_width_, label="SVM")

axs.bar(X_axis - bar_width_, EVI_regular_crop_summary_L_LSD_2cropped.KNN_EVI_regular_preds, 
        color ='red', width = bar_width_, label="kNN")

axs.bar(X_axis, EVI_regular_crop_summary_L_LSD_2cropped.DL_EVI_regular_prob_point4,
        color ='k', width = bar_width_, label="DL")

axs.bar(X_axis + bar_width_, EVI_regular_crop_summary_L_LSD_2cropped.RF_EVI_regular_preds,
        color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, EVI_regular_crop_summary_L_LSD_2cropped.CropTyp)

axs.set_ylabel("double-cropped (%)")
axs.set_xlabel("crop type")
axs.set_title("4-step EVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");

file_name = plot_dir + "regular_EVI_cropWise_precent.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(NDVI_regular_crop_summary_L_LSD_2cropped.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(NDVI_regular_crop_summary_L_LSD_2cropped.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, NDVI_regular_crop_summary_L_LSD_2cropped.SVM_NDVI_regular_preds, 
        color ='dodgerblue', width = bar_width_, label="SVM")

axs.bar(X_axis - bar_width_, NDVI_regular_crop_summary_L_LSD_2cropped.KNN_NDVI_regular_preds,
        color ='red', width = bar_width_, label="kNN")

axs.bar(X_axis, NDVI_regular_crop_summary_L_LSD_2cropped.DL_NDVI_regular_prob_point9, 
        color ='k', width = bar_width_, label="DL")

axs.bar(X_axis + bar_width_, NDVI_regular_crop_summary_L_LSD_2cropped.RF_NDVI_regular_preds, 
        color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, NDVI_regular_crop_summary_L_LSD_2cropped.CropTyp)

axs.set_ylabel("double-cropped (%)")
axs.set_xlabel("crop type")
axs.set_title("4-step NDVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");

file_name = plot_dir + "regular_NDVI_cropWise_precent.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()

# %%

# %%

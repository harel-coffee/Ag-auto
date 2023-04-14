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
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
pred_dir_base = "/Users/hn/Documents/01_research_data/NASA/RegionalStatData/"
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
SVM = pd.DataFrame()
KNN = pd.DataFrame()
DL  = pd.DataFrame()
RF  = pd.DataFrame()

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
SVM = pd.DataFrame()
KNN = pd.DataFrame()
DL  = pd.DataFrame()
RF  = pd.DataFrame()

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
SVM = pd.DataFrame()
KNN = pd.DataFrame()
DL  = pd.DataFrame()
RF  = pd.DataFrame()

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
SVM = pd.DataFrame()
KNN = pd.DataFrame()
DL  = pd.DataFrame()
RF  = pd.DataFrame()

NDVI_regular_DL_prob = "prob_point9"
for a_file in NDVI_regular:
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
all_preds = pd.merge(all_preds, EVI_SG_preds, on="ID", how='left')
all_preds = pd.merge(all_preds, NDVI_SG_preds, on="ID", how='left')

all_preds.head(2)

# %%
SF_data = SF_data[["ID", "CropTyp", "Acres", "ExctAcr", "Irrigtn", "LstSrvD", "DataSrc", "county"]]
SF_data = nc.filter_out_nonIrrigated(SF_data)

all_preds =          pd.merge(NDVI_regular_preds, SF_data, on="ID", how='left')
EVI_SG_preds =       pd.merge(EVI_SG_preds,       SF_data, on="ID", how='left')
NDVI_SG_preds =      pd.merge(NDVI_SG_preds,      SF_data, on="ID", how='left')
EVI_regular_preds =  pd.merge(EVI_regular_preds,  SF_data, on="ID", how='left')
NDVI_regular_preds = pd.merge(NDVI_regular_preds, SF_data, on="ID", how='left')

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

# EVI_SG_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

# out_name = pred_dir_base + "EVI_SG_summary.csv"
EVI_SG_summary.to_csv(out_name, index = False)

EVI_SG_summary.round()

# %%
NDVI_SG_summary = pd.DataFrame(columns=list(NDVI_SG_preds.columns[1:5]))
NDVI_SG_summary[NDVI_SG_summary.columns[0]] = NDVI_SG_preds.groupby([NDVI_SG_summary.columns[0]])['ExctAcr'].sum()
NDVI_SG_summary[NDVI_SG_summary.columns[1]] = NDVI_SG_preds.groupby([NDVI_SG_summary.columns[1]])['ExctAcr'].sum()
NDVI_SG_summary[NDVI_SG_summary.columns[2]] = NDVI_SG_preds.groupby([NDVI_SG_summary.columns[2]])['ExctAcr'].sum()
NDVI_SG_summary[NDVI_SG_summary.columns[3]] = NDVI_SG_preds.groupby([NDVI_SG_summary.columns[3]])['ExctAcr'].sum()
NDVI_SG_summary.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")

out_name = pred_dir_base + "NDVI_SG_summary.csv"
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

out_name = pred_dir_base + "NDVI_regularular_summary.csv"
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

# %%

# %%

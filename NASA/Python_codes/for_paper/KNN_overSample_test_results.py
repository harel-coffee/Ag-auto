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

from datetime import date, datetime
import time

import random
from random import seed
from random import random

import os, os.path
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from pylab import imshow
import pickle
import h5py
import sys

# %%
nasa_data_base_dir = "/Users/hn/Documents/01_research_data/NASA/"
KNN_oversample_testResult_dir = nasa_data_base_dir + "ML_data_Oct17/KNN_results/overSample/"

# %%
CSV_files = [x for x in os.listdir(KNN_oversample_testResult_dir) if x.endswith(".csv")]
CSV_files=sorted(CSV_files)

# %%

# %%
all_results = pd.DataFrame(columns=["type", "A1P1", "A2P2", "A1P2", "A2P1", "error"], index=range(24))
curr_row=0
for a_res in CSV_files:
    a_res_df = pd.read_csv(KNN_oversample_testResult_dir+a_res)
    
    true_single_predicted_single=0
    true_single_predicted_double=0

    true_double_predicted_single=0
    true_double_predicted_double=0

    for index_ in a_res_df.index:
        curr_vote=list(a_res_df[a_res_df.index==index_].Vote)[0]
        curr_predict=list(a_res_df[a_res_df.index==index_].prediction)[0]
        if curr_vote==curr_predict:
            if curr_vote==1: 
                true_single_predicted_single+=1
            else:
                true_double_predicted_double+=1
        else:
            if curr_vote==1:
                true_single_predicted_double+=1
            else:
                true_double_predicted_single+=1

    confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                                 index=range(2))
    confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
    confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
    confus_tbl_test['Predict_Single']=0
    confus_tbl_test['Predict_Double']=0

    confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
    confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
    confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
    confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
    
    error = confus_tbl_test.iloc[1, 1]+confus_tbl_test.iloc[0, 2]
    
    a_res_split = a_res.split("_")
    smooth_method = a_res_split[1]
    VI_idx = a_res_split[2]
    SR = a_res_split[6][-1]
    
    # print (f"|{VI_idx=:}, {smooth_method=:}, and {SR=:}|")
    print (f"|{VI_idx}, {smooth_method}, {SR=:}|")
    print(f"error={error:=^10}")
    print (confus_tbl_test)
    variable=""
    print(f"{variable:=^100}")
    
    all_results.loc[curr_row, "type"] = VI_idx + "-" + smooth_method + "-" + SR
    all_results.loc[curr_row, "A1P1"] = confus_tbl_test.iloc[0, 1]
    all_results.loc[curr_row, "A2P2"] = confus_tbl_test.iloc[1, 2]
    all_results.loc[curr_row, "A1P2"] = confus_tbl_test.iloc[0, 2]
    all_results.loc[curr_row, "A2P1"] = confus_tbl_test.iloc[1, 1]
    all_results.loc[curr_row, "error"] = error
    curr_row+=1

# %%
all_results

# %%
print (CSV_files[0])
print (CSV_files[1])
CSV_files_0 = pd.read_csv(KNN_oversample_testResult_dir+CSV_files[0])
CSV_files_1 = pd.read_csv(KNN_oversample_testResult_dir+CSV_files[1])

# %%
wide_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"
EVI_SG_SR3=pd.read_csv(wide_dir+"EVI_SG_wide_train80_split_2Bconsistent_Oct17_overSample3.csv")
EVI_SG_SR4=pd.read_csv(wide_dir+"EVI_SG_wide_train80_split_2Bconsistent_Oct17_overSample4.csv")

# %%
print (EVI_SG_SR3.shape)
print (EVI_SG_SR4.shape)

# %%
CSV_files_0.shape

# %%

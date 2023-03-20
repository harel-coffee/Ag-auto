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
import scipy, scipy.signal

from datetime import date
import time

from random import seed
from random import random

import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import h5py
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/06_SOS_tables/"

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
sorted(meta.county.unique())

# %%
walla = meta[meta.county=="Walla Walla"]
walla.ID

# %%

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
ground_truth_labels.shape

# %% [markdown]
# # Test set (the 20%)

# %%
test_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
testset = pd.read_csv(test_set_dir+"test20_split_expertLabels_2Bconsistent.csv")
ground_truth_labels_test = ground_truth_labels[ground_truth_labels.ID.isin(list(testset.ID))]
ground_truth_labels_test.shape

# %%
all_files = os.listdir(data_dir)

NamePattern="irr_NoNASS_SurvCorrect"
files_with_3_filters = [x for x in all_files if NamePattern in x]

# %%
VI_indeksss = ["NDVI", "EVI"]
NDVI_thresholds = [3, 4, 5]

VI_indeks="EVI"
NDVI_threshold=3

for VI_indeks in VI_indeksss:
    for NDVI_threshold  in NDVI_thresholds:
        all_data = pd.DataFrame()

        for fileName in files_with_3_filters:
            if VI_indeks+str(NDVI_threshold) in fileName:
                a=pd.read_csv(data_dir + fileName)
                a['human_system_start_time'] = pd.to_datetime(a['human_system_start_time'])
                curr_year = int(fileName.split("_")[2][-4:])
                a = a[a['human_system_start_time'].dt.year == curr_year].copy()
                a = a[["ID", "season_count"]]

                all_data=pd.concat([all_data, a])

        all_data = all_data[all_data.ID.isin(list(ground_truth_labels_test.ID.unique()))].copy()
        all_data.drop_duplicates(inplace=True)
        all_data.reset_index(drop=True, inplace=True)
        print ("==================================================================================================")
        single_season=all_data[all_data.season_count<2]
        two_seasons=all_data[all_data.season_count>=2]
        print ("this must be 269:", str(len(single_season)+len(two_seasons)))
        
        evalHelp = pd.merge(ground_truth_labels_test, all_data, on=['ID'], how='left')
        true_single_predicted_single = evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_single = true_single_predicted_single[true_single_predicted_single.season_count<2]


        true_double_predicted_double = evalHelp[evalHelp.Vote>=2].copy()
        true_double_predicted_double = true_double_predicted_double[true_double_predicted_double.season_count>=2]

        true_double_predicted_single=evalHelp[evalHelp.Vote==2].copy()
        true_double_predicted_single = true_double_predicted_single[true_double_predicted_single.season_count<2]

        true_single_predicted_double=evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_double = true_single_predicted_double[true_single_predicted_double.season_count>=2]
        
        ######        ######        ######
        true_double_predicted_2=evalHelp[evalHelp.Vote==2].copy()
        true_double_predicted_2 = true_double_predicted_2[true_double_predicted_2.season_count>=2]

        true_single_predicted_1=evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_1= true_single_predicted_1[true_single_predicted_1.season_count<2]
        ######        ######        ######

        balanced_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                                index=range(2))
        balanced_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
        balanced_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
        balanced_confus_tbl_test['Predict_Single']=0
        balanced_confus_tbl_test['Predict_Double']=0

        balanced_confus_tbl_test.loc[0, "Predict_Single"]=len(true_single_predicted_single)
        balanced_confus_tbl_test.loc[0, "Predict_Double"]=len(true_single_predicted_double)
        balanced_confus_tbl_test.loc[1, "Predict_Single"]=len(true_double_predicted_single)
        balanced_confus_tbl_test.loc[1, "Predict_Double"]=len(true_double_predicted_double)
        
        print ("VI_indeks: " + VI_indeks + ", NDVI_threshold: " + str(NDVI_threshold))
        print (balanced_confus_tbl_test)
        print ("")
        _dc = np.abs(balanced_confus_tbl_test.loc[0, "Predict_Double"]-balanced_confus_tbl_test.loc[1, "Predict_Single"])
        print ("count difference is " + str(_dc))
        acr_diff=np.abs(true_double_predicted_single.ExctAcr.sum()-true_single_predicted_double.ExctAcr.sum())
        print ("acr difference is "+str(acr_diff))
        print ("true_double_predicted_single.ExctAcr", true_double_predicted_single.ExctAcr.sum())
        print ("true_single_predicted_double.ExctAcr", true_single_predicted_double.ExctAcr.sum())
        print ()
        
        print ("true_double_predicted_2.ExctAcr", true_double_predicted_2.ExctAcr.sum())
        print ("true_single_predicted_1.ExctAcr", true_single_predicted_1.ExctAcr.sum())


# %%

# %% [markdown]
# # Drop Walla Walla

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
ground_truth_labels=ground_truth_labels[ground_truth_labels.county!="Walla Walla"]

# %% [markdown]
# # Test Set (20%)

# %%
test_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
testset = pd.read_csv(test_set_dir+"test20_split_expertLabels_2Bconsistent.csv")

ground_truth_labels_test = ground_truth_labels[ground_truth_labels.ID.isin(list(testset.ID))]
ground_truth_labels_test.shape

# %%
all_files = os.listdir(data_dir)

NamePattern="irr_NoNASS_SurvCorrect"
files_with_3_filters = [x for x in all_files if NamePattern in x]

# %%

# %%
VI_indeksss = ["NDVI", "EVI"]
NDVI_thresholds = [3, 4, 5]

VI_indeks="EVI"
NDVI_threshold=3

for VI_indeks in VI_indeksss:
    for NDVI_threshold  in NDVI_thresholds:
        all_data = pd.DataFrame()

        for fileName in files_with_3_filters:
            if VI_indeks+str(NDVI_threshold) in fileName:
                a=pd.read_csv(data_dir + fileName)
                a['human_system_start_time'] = pd.to_datetime(a['human_system_start_time'])
                curr_year = int(fileName.split("_")[2][-4:])
                a = a[a['human_system_start_time'].dt.year == curr_year].copy()
                a = a[["ID", "season_count"]]

                all_data=pd.concat([all_data, a])

        all_data = all_data[all_data.ID.isin(list(ground_truth_labels_test.ID.unique()))].copy()
        all_data.drop_duplicates(inplace=True)
        all_data.reset_index(drop=True, inplace=True)
        print ("==================================================================================================")
        single_season=all_data[all_data.season_count<2]
        two_seasons=all_data[all_data.season_count>=2]
        print ("this must be 269:", str(len(single_season)+len(two_seasons)))
        
        evalHelp = pd.merge(ground_truth_labels_test, all_data, on=['ID'], how='left')
        true_single_predicted_single = evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_single = true_single_predicted_single[true_single_predicted_single.season_count<2]


        true_double_predicted_double = evalHelp[evalHelp.Vote>=2].copy()
        true_double_predicted_double = true_double_predicted_double[true_double_predicted_double.season_count>=2]

        true_double_predicted_single=evalHelp[evalHelp.Vote==2].copy()
        true_double_predicted_single = true_double_predicted_single[true_double_predicted_single.season_count<2]

        true_single_predicted_double=evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_double = true_single_predicted_double[true_single_predicted_double.season_count>=2]


        balanced_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                                index=range(2))
        balanced_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
        balanced_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
        balanced_confus_tbl_test['Predict_Single']=0
        balanced_confus_tbl_test['Predict_Double']=0

        balanced_confus_tbl_test.loc[0, "Predict_Single"]=len(true_single_predicted_single)
        balanced_confus_tbl_test.loc[0, "Predict_Double"]=len(true_single_predicted_double)
        balanced_confus_tbl_test.loc[1, "Predict_Single"]=len(true_double_predicted_single)
        balanced_confus_tbl_test.loc[1, "Predict_Double"]=len(true_double_predicted_double)
        
        print ("VI_indeks: " + VI_indeks + ", NDVI_threshold: " + str(NDVI_threshold))
        print (balanced_confus_tbl_test)
        print ("")
        _dc = np.abs(balanced_confus_tbl_test.loc[0, "Predict_Double"]-balanced_confus_tbl_test.loc[1, "Predict_Single"])
        print ("count difference is " + str(_dc))
        acr_diff=np.abs(true_double_predicted_single.ExctAcr.sum()-true_single_predicted_double.ExctAcr.sum())
        print ("acr difference is "+str(acr_diff))
        print ("true_double_predicted_single.ExctAcr", true_double_predicted_single.ExctAcr.sum())
        print ("true_single_predicted_double.ExctAcr", true_single_predicted_double.ExctAcr.sum())
        print ()

# %%

# %% [markdown]
# # Sentinel

# %%
sent_dir = "/Users/hn/Documents/01_research_data/NASA/Sentinel/"

# %%

# %%
ground_truth_labels=ground_truth_labels[ground_truth_labels.county!="Walla Walla"]
ground_truth_labels_test = ground_truth_labels[ground_truth_labels.ID.isin(list(testset.ID))]
ground_truth_labels_test.shape

# %%
# fileNames = ["extended_all_fields_seasonCounts_noFilter_SEOS3", "extended_all_fields_seasonCounts_noFilter_SEOS4",
#              "extended_all_fields_seasonCounts_noFilter_SEOS5"]
# sentinel_data=pd.DataFrame()
# for a_file in fileNames:
#     a=pd.read_csv(sent_dir+"05_01_allFields_SeasonCounts/" + a_file+".csv")
#     a=a[a.SG_params==73]
#     sentinel_data=pd.concat([sentinel_data, a])

# %%
VI_indeksss = ["NDVI", "EVI"]
years=[2016, 2017, 2018]
NDVI_thresholds = [3, 4, 5]
Name_pattern = "win7_Order3"
folder_prePattern = "2Yrs_tbl_reg_fineGranular_SOS"

# %%
VI_indeks="NDVI"
NDVI_threshold=3
all_data=pd.DataFrame()

# %%
for VI_indeks in VI_indeksss:
    for NDVI_threshold in NDVI_thresholds:
        all_data=pd.DataFrame()
        for year in years:
            folder_name = folder_prePattern+str(NDVI_threshold)+"_EOS"+str(NDVI_threshold)
            data_dir = sent_dir+folder_name+"/"
            file_list = os.listdir(data_dir)
            file_list_SG73 = [x for x in file_list if Name_pattern in x]
            file_list_SG73 = [x for x in file_list_SG73 if VI_indeks in x]

            if year==2016:
                cnty1_names = [x for x in file_list_SG73 if "Adams_2016" in x]
                cnty2_names = [x for x in file_list_SG73 if "Benton_2016" in x]
                finalFileNames = cnty1_names+cnty2_names
            elif year==2017:
                finalFileNames= [x for x in file_list_SG73 if "Grant_2017" in x]
            elif year==2018:
                cnty1_names = [x for x in file_list_SG73 if "Franklin_2018" in x]
                cnty2_names = [x for x in file_list_SG73 if "Yakima_2018" in x]
                finalFileNames = cnty1_names+cnty2_names

            # print(VI_indeks, year, NDVI_threshold, finalFileNames)
            aIndeks_all_data=pd.DataFrame()
            for a_file in finalFileNames:
                a=pd.read_csv(data_dir+a_file)
                a['human_system_start_time'] = pd.to_datetime(a['human_system_start_time'])
                aIndeks_all_data=pd.concat([aIndeks_all_data, a])
            all_data=pd.concat([all_data, aIndeks_all_data])
            all_data = all_data[all_data.ID.isin(list(ground_truth_labels_test.ID.unique()))].copy()
            all_data=all_data[["ID", "season_count"]]
            all_data.drop_duplicates(inplace=True)
            all_data.reset_index(drop=True, inplace=True)

        print ("==================================================================================================")
        single_season=all_data[all_data.season_count<2]
        two_seasons=all_data[all_data.season_count>=2]
        print ("this must be 248:", str(len(single_season)+len(two_seasons)))
        
        evalHelp = pd.merge(ground_truth_labels_test, all_data, on=['ID'], how='left')
        true_single_predicted_single = evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_single = true_single_predicted_single[true_single_predicted_single.season_count<2]


        true_double_predicted_double = evalHelp[evalHelp.Vote>=2].copy()
        true_double_predicted_double = true_double_predicted_double[true_double_predicted_double.season_count>=2]

        true_double_predicted_single=evalHelp[evalHelp.Vote==2].copy()
        true_double_predicted_single = true_double_predicted_single[true_double_predicted_single.season_count<2]

        true_single_predicted_double=evalHelp[evalHelp.Vote==1].copy()
        true_single_predicted_double = true_single_predicted_double[true_single_predicted_double.season_count>=2]


        balanced_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                                index=range(2))
        balanced_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
        balanced_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
        balanced_confus_tbl_test['Predict_Single']=0
        balanced_confus_tbl_test['Predict_Double']=0

        balanced_confus_tbl_test.loc[0, "Predict_Single"]=len(true_single_predicted_single)
        balanced_confus_tbl_test.loc[0, "Predict_Double"]=len(true_single_predicted_double)
        balanced_confus_tbl_test.loc[1, "Predict_Single"]=len(true_double_predicted_single)
        balanced_confus_tbl_test.loc[1, "Predict_Double"]=len(true_double_predicted_double)

        print ("VI_indeks: " + VI_indeks + ", NDVI_threshold: " + str(NDVI_threshold))
        print (balanced_confus_tbl_test)
        print ("")
        _dc = np.abs(balanced_confus_tbl_test.loc[0, "Predict_Double"]-balanced_confus_tbl_test.loc[1, "Predict_Single"])
        print ("count difference is " + str(_dc))
        acr_diff=np.abs(true_double_predicted_single.ExctAcr.sum()-true_single_predicted_double.ExctAcr.sum())
        print ("acr difference is "+str(acr_diff))
        print ("true_double_predicted_single.ExctAcr", true_double_predicted_single.ExctAcr.sum())
        print ("true_single_predicted_double.ExctAcr", true_single_predicted_double.ExctAcr.sum())
        print ()


# %%
all_data.shape

# %%
adam_2016=pd.read_csv(data_dir+"Adams_2016_regular_EVI_SG_win7_Order3.csv")
adam_2017=pd.read_csv(data_dir+"Adams_2017_regular_EVI_SG_win7_Order3.csv")
adam_2018=pd.read_csv(data_dir+"Adams_2018_regular_EVI_SG_win7_Order3.csv")

# %%
all_data[all_data.ID=="102173_WSDA_SF_2018"]

# %%
all_data.county.unique()

# %%
all_data = all_data[all_data.ID.isin(list(ground_truth_labels_test.ID.unique()))].copy()
len(all_data.ID.unique())

# %%
Adams = all_data[all_data.county=="Franklin"]
Adams['human_system_start_time'].dt.year.unique()

# %%

# %%

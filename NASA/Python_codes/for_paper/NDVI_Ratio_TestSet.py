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
import datetime
import time
import scipy
import scipy.signal
import os, os.path
import sys
from datetime import date, datetime
# from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report
# from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %% [markdown]
# ### Directories

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
train_TS_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"

# %% [markdown]
# ### Metadata

# %%
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
ground_truth_labels = pd.read_csv(ML_data_folder+"groundTruth_labels_Oct17_2022.csv")
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin(list(meta.ID.unique()))].copy()
ground_truth_labels = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
print ("ground_truth_labels shape is [{}].".format(ground_truth_labels.shape))
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print ("Minimum size is [{}].".format(round(ground_truth_labels.ExctAcr.min(), 3)))
print ("Number of unique fields are [{}].".format( len(ground_truth_labels.ID.unique())))
ground_truth_labels.head(2)

# %%
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")

# %% [markdown]
# ### Parameters

# %%
# onset_cut=offset_cut=0.3 # 0.3, 0.4, 0.5
# We already have done this before and all other MLs are based on them.
# SG_win_size=7 # 5, 7
# SG_order=3 

VI_idx      = "EVI" # ["EVI", "NDVI"]
smooth_type = "SG"  # ["SG", "regular"]

# %%
all_season_preds = pd.DataFrame(data = None, 
                                index = np.arange(len(train80.ID.unique())), 
                                columns = ["ID", "EVI_SG_season_count3",
                                                 "EVI_SG_season_count4",
                                                 "EVI_SG_season_count5",
                                                 "EVI_SG_season_count6",
                                                 "EVI_SG_season_count7",
                                           
                                                 "NDVI_SG_season_count3",
                                                 "NDVI_SG_season_count4",
                                                 "NDVI_SG_season_count5",
                                                 "NDVI_SG_season_count6",
                                                 "NDVI_SG_season_count7",

                                                 "EVI_regular_season_count3",
                                                 "EVI_regular_season_count4",
                                                 "EVI_regular_season_count5",
                                                 "EVI_regular_season_count6",
                                                 "EVI_regular_season_count7",

                                                 "NDVI_regular_season_count3",
                                                 "NDVI_regular_season_count4",
                                                 "NDVI_regular_season_count5",
                                                 "NDVI_regular_season_count6",
                                                 "NDVI_regular_season_count7"])

all_season_preds["ID"] = train80.ID.unique()

# %%
VI_indices = ["EVI", "NDVI"]
smooth_types = ["SG", "regular"]

# for threshold in [3, 4, 5, 5.5, 6, 7]:
thresholds = [3, 4, 5, 6, 7]
for threshold in thresholds:
    onset_cut=offset_cut=(threshold/10.0)
    for VI_idx in VI_indices:
        for smooth_type in smooth_types:
            print (threshold, VI_idx, smooth_type)
            if smooth_type=="SG":
                train_TS_dir = train_TS_dir_base + "05_SG_TS/"
            else:
                train_TS_dir = train_TS_dir_base + "04_regularized_TS/"

            file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

            all_TS=pd.DataFrame()
            for file in file_names:
                curr_file=pd.read_csv(train_TS_dir + file)
                curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])

                # These data are for 3 years. The middle one is the correct one
                all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
                if len(all_years)==3 or len(all_years)==2:
                    proper_year = all_years[1]
                elif len(all_years)==1:
                    proper_year = all_years[0]

                curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
                all_TS=pd.concat([all_TS, curr_file])

            all_TS = all_TS[all_TS.ID.isin(list(ground_truth_labels.ID.unique()))].copy()

            # The following is not neccessary because we already have only large fields in ground_truth_labels.
            # all_TS = all_TS[all_TS.ID.isin(list(meta.ID.unique()))].copy()

            all_TS.reset_index(drop=True, inplace=True)
            all_TS['human_system_start_time'] = pd.to_datetime(all_TS["human_system_start_time"])
            train_TS = all_TS[all_TS.ID.isin(train80.ID.unique())].copy()
            # print ("all_TS shape is {}".format(all_TS.shape))

            SEOS_output_columns = ["ID", "human_system_start_time", VI_idx, 
                                    VI_idx + "_ratio", "SOS", "EOS", "season_count"]
            #
            # The reason I am multiplying len(a_df) by 4 is that we can have at least two
            # seasons which means 2 SOS and 2 EOS. So, at least 4 rows are needed.
            #
            all_poly_and_SEOS = pd.DataFrame(data = None, index = np.arange(4*len(train_TS)), 
                                             columns = SEOS_output_columns)
            counter = 0
            pointer_SEOS_tab = 0

            field_IDs = train_TS.ID.unique()

            for an_ID in field_IDs:
                curr_field = train_TS[train_TS['ID']==an_ID].copy()
                curr_field.reset_index(drop=True, inplace=True)
                curr_field.sort_values(by=['human_system_start_time'], inplace=True)
                curr_field["doy"] = curr_field['human_system_start_time'].dt.dayofyear

                curr_YR=str(curr_field.human_system_start_time[10].year)
                y_orchard = curr_field[curr_field['human_system_start_time'] >= pd.to_datetime("05-01-"+curr_YR)]
                y_orchard = y_orchard[y_orchard['human_system_start_time'] <= pd.to_datetime("11-01-"+curr_YR)]
                y_orchard_range = max(y_orchard[VI_idx]) - min(y_orchard[VI_idx])
                if y_orchard_range > 0.3:
                    #######################################################################
                    ###
                    ###             find SOS and EOS, and add them to the table
                    ###
                    #######################################################################
                    fine_granular_table = nc.create_calendar_table(SF_year=curr_YR)
                    fine_granular_table = pd.merge(fine_granular_table, curr_field, 
                                                    on=['human_system_start_time', 'doy'], how='left')

                    # We need to fill the NAs that are created 
                    # because they were not created in fine_granular_table
                    fine_granular_table["ID"] = curr_field["ID"].unique()[0]

                    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
                    # uses as indicator for missing values
                    fine_granular_table.fillna(value={VI_idx:-1.5}, inplace=True)

                    fine_granular_table = nc.fill_theGap_linearLine(a_regularized_TS = fine_granular_table,\
                                                                    V_idx=VI_idx)

                    fine_granular_table = nc.addToDF_SOS_EOS_White(pd_TS = fine_granular_table, 
                                                                   VegIdx = VI_idx, 
                                                                   onset_thresh = onset_cut, # onset_cut
                                                                   offset_thresh = offset_cut) # offset_cut
                    ##
                    ## Kill false detected seasons 
                    ##
                    fine_granular_table=nc.Null_SOS_EOS_by_DoYDiff(pd_TS=fine_granular_table, \
                                                                   min_season_length=40)
                    #
                    # extract the SOS and EOS rows 
                    #
                    SEOS = fine_granular_table[(fine_granular_table['SOS']!=0) | fine_granular_table['EOS']!=0]
                    SEOS = SEOS.copy()

                    # SEOS = SEOS.reset_index() # not needed really
                    SOS_tb = fine_granular_table[fine_granular_table['SOS'] != 0]

                    if len(SOS_tb) >= 2:
                        SEOS["season_count"] = len(SOS_tb)
                        # re-order columns of SEOS so they match!!!
                        SEOS = SEOS[all_poly_and_SEOS.columns]
                        all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
                        pointer_SEOS_tab += len(SEOS)
                    else:
                        # re-order columns of fine_granular_table so they match!!!
                        fine_granular_table["season_count"] = 1
                        fine_granular_table = fine_granular_table[all_poly_and_SEOS.columns]
                        aaa = fine_granular_table.iloc[0].values.reshape(1, len(fine_granular_table.iloc[0]))
                        all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                        pointer_SEOS_tab += 1
                else: # here are potentially apples, cherries, etc.
                    # we did not add EVI_ratio, SOS, and EOS. So, we are missing these
                    # columns in the data frame. So, use 666 as proxy
                    aaa = np.append(curr_field.iloc[0][:3], [666, 666, 666, 1])
                    aaa = aaa.reshape(1, len(aaa))
                    all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                    pointer_SEOS_tab += 1

            just_season_counts = all_poly_and_SEOS[["ID", "season_count"]].copy()
            just_season_counts.drop_duplicates(inplace=True)
            just_season_counts.reset_index(drop=True, inplace=True)
            colName = VI_idx + "_" + smooth_type + "_season_count" + str(threshold)
            all_season_preds[colName] = just_season_counts["season_count"]

# %%
all_season_preds = pd.merge(all_season_preds, ground_truth_labels, on=["ID"], how="left")
all_season_preds.head(2)

# %% [markdown]
# ### Convert everything to 2 if number of seasons is more than 2

# %%
predict_cols = all_season_preds.columns[1:len(list(all_season_preds.columns))-8]
for a_col in predict_cols:
    all_season_preds.loc[all_season_preds[a_col]>2, a_col]=2

# %% [markdown]
# # Analysis
#
# **Accuracy** is not the best way to measure and compare. $\tau=1$ produces the best result! Everything will be labeled as single cropped. So, I look at F-1 score and macro/micro accuracy; weighted by class numbers!

# %%
EVI_SGs  = ["EVI_SG_season_count3", "EVI_SG_season_count4", "EVI_SG_season_count5", 
            "EVI_SG_season_count6", "EVI_SG_season_count7"]

NDVI_SGs = ["NDVI_SG_season_count3", "NDVI_SG_season_count4","NDVI_SG_season_count5", 
            "NDVI_SG_season_count6", "NDVI_SG_season_count7"]

EVI_regulars  = ["EVI_regular_season_count3", "EVI_regular_season_count4", "EVI_regular_season_count5", 
                 "EVI_regular_season_count6", "EVI_regular_season_count7"]

NDVI_regulars = ["NDVI_regular_season_count3", "NDVI_regular_season_count4", "NDVI_regular_season_count5", 
                 "NDVI_regular_season_count6", "NDVI_regular_season_count7"]

col_set = EVI_SGs + NDVI_SGs + EVI_regulars + NDVI_regulars

# %%
acc_table = pd.DataFrame(data = None, 
                         index = np.arange(2), 
                         columns = ["stat"]+col_set)
acc_table["stat"] = ["Accuracy", "Error_Count"]

field_count = len(all_season_preds.ID.unique())
for a_col in col_set:
    error=sum(all_season_preds["Vote"]!=all_season_preds[a_col])
    acc = sum(all_season_preds["Vote"]==all_season_preds[a_col])/field_count
    acc_table[a_col] = [acc, error]
    
acc_table

# %%
col="EVI_SG_season_count3"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_SG_season_count4"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_SG_season_count5"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_SG_season_count6"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_SG_season_count7"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

# col="EVI_SG_season_count8"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# col="EVI_SG_season_count9"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# %%
col="NDVI_SG_season_count3"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_SG_season_count4"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_SG_season_count5"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_SG_season_count6"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_SG_season_count7"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

# col="NDVI_SG_season_count8"
# print (col)
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# col="NDVI_SG_season_count9"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# %%
col="EVI_regular_season_count3"
print (col)    
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_regular_season_count4"
print (col)    
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_regular_season_count5"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_regular_season_count6"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))

print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="EVI_regular_season_count7"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

# col="EVI_regular_season_count8"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# col="EVI_regular_season_count9"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# %%
col="NDVI_regular_season_count3"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_regular_season_count4"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_regular_season_count5"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_regular_season_count6"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

col="NDVI_regular_season_count7"
print (col)
print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
print ("________________________________________________________________________________________")

# col="NDVI_regular_season_count8"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# col="NDVI_regular_season_count9"
# print (col)
# print(confusion_matrix(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print(classification_report(all_season_preds["Vote"].astype(int), all_season_preds[col].astype(int)))
# print ("________________________________________________________________________________________")

# %%

# %%
# acc_table[["stat"]+ NDVI_SGs]

# %%
# acc_table[["stat"]+ EVI_regulars]

# %%
# acc_table[["stat"]+ NDVI_regulars]

# %% [markdown]
# ### Test Set
#
# It seems $\tau=XXX$ is winner in all cases. So, let us use that on test set and report the result!

# %%
# x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
#                         "_wide_test20_split_2Bconsistent_Oct17.csv")
# print (len(x_test_df.ID.unique()))
# x_test_df.head(2)

# %%
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")
test20.shape

# %%
test_TS_dir_base = train_TS_dir_base
test_season_preds = pd.DataFrame(data = None, 
                                index = np.arange(len(test20.ID.unique())), 
                                columns = ["ID", "EVI_SG_season_count5",
                                                 "NDVI_SG_season_count5",
                                                 "EVI_regular_season_count5",
                                                 "NDVI_regular_season_count5"])
test_season_preds["ID"] = test20.ID.unique()

VI_indices = ["EVI", "NDVI"]
smooth_types = ["SG", "regular"]

for threshold in [5]:
    onset_cut=offset_cut=(threshold/10.0)
    for VI_idx in VI_indices:
        for smooth_type in smooth_types:
            print (threshold, VI_idx, smooth_type)
            
            if smooth_type=="SG":
                test_TS_dir = test_TS_dir_base + "05_SG_TS/"
            else:
                test_TS_dir = test_TS_dir_base + "04_regularized_TS/"

            file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
                          smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

            all_TS=pd.DataFrame()

            for file in file_names:
                curr_file=pd.read_csv(test_TS_dir + file)
                curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])

                # These data are for 3 years. The middle one is the correct one
                all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
                if len(all_years)==3 or len(all_years)==2:
                    proper_year = all_years[1]
                elif len(all_years)==1:
                    proper_year = all_years[0]

                curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
                all_TS=pd.concat([all_TS, curr_file])

            all_TS = all_TS[all_TS.ID.isin(list(ground_truth_labels.ID.unique()))].copy()

            all_TS.reset_index(drop=True, inplace=True)
            all_TS['human_system_start_time'] = pd.to_datetime(all_TS["human_system_start_time"])
            test_TS = all_TS[all_TS.ID.isin(test20.ID.unique())].copy()
            SEOS_output_columns = ["ID", "human_system_start_time", VI_idx, 
                                    VI_idx + "_ratio", "SOS", "EOS", "season_count"]
            #
            # The reason I am multiplying len(a_df) by 4 is that we can have at least two
            # seasons which means 2 SOS and 2 EOS. So, at least 4 rows are needed.
            #
            all_poly_and_SEOS = pd.DataFrame(data = None, index = np.arange(4*len(test_TS)), 
                                             columns = SEOS_output_columns)
            counter = 0
            pointer_SEOS_tab = 0

            field_IDs = test_TS.ID.unique()

            for an_ID in field_IDs:
                curr_field = test_TS[test_TS['ID']==an_ID].copy()
                curr_field.reset_index(drop=True, inplace=True)
                curr_field.sort_values(by=['human_system_start_time'], inplace=True)
                curr_field["doy"] = curr_field['human_system_start_time'].dt.dayofyear

                curr_YR=str(curr_field.human_system_start_time[10].year)
                y_orchard = curr_field[curr_field['human_system_start_time'] >= pd.to_datetime("05-01-"+curr_YR)]
                y_orchard = y_orchard[y_orchard['human_system_start_time'] <= pd.to_datetime("11-01-"+curr_YR)]
                y_orchard_range = max(y_orchard[VI_idx]) - min(y_orchard[VI_idx])
                if y_orchard_range > 0.3:
                    #######################################################################
                    ###             find SOS and EOS, and add them to the table
                    #######################################################################
                    fine_granular_table = nc.create_calendar_table(SF_year=curr_YR)
                    fine_granular_table = pd.merge(fine_granular_table, curr_field, 
                                                    on=['human_system_start_time', 'doy'], how='left')

                    # We need to fill the NAs that are created 
                    # because they were not created in fine_granular_table
                    fine_granular_table["ID"] = curr_field["ID"].unique()[0]

                    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
                    # uses as indicator for missing values
                    fine_granular_table.fillna(value={VI_idx:-1.5}, inplace=True)

                    fine_granular_table = nc.fill_theGap_linearLine(a_regularized_TS = fine_granular_table,\
                                                                    V_idx=VI_idx)

                    fine_granular_table = nc.addToDF_SOS_EOS_White(pd_TS = fine_granular_table, 
                                                                   VegIdx = VI_idx, 
                                                                   onset_thresh = onset_cut, # onset_cut
                                                                   offset_thresh = offset_cut) # offset_cut
                    ##  Kill false detected seasons 
                    fine_granular_table=nc.Null_SOS_EOS_by_DoYDiff(pd_TS=fine_granular_table, \
                                                                   min_season_length=40)
                    # extract the SOS and EOS rows 
                    SEOS = fine_granular_table[(fine_granular_table['SOS']!=0) | fine_granular_table['EOS']!=0]
                    SEOS = SEOS.copy()
                    # SEOS = SEOS.reset_index() # not needed really
                    SOS_tb = fine_granular_table[fine_granular_table['SOS'] != 0]
                    if len(SOS_tb) >= 2:
                        SEOS["season_count"] = len(SOS_tb)
                        # re-order columns of SEOS so they match!!!
                        SEOS = SEOS[all_poly_and_SEOS.columns]
                        all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
                        pointer_SEOS_tab += len(SEOS)
                    else:
                        # re-order columns of fine_granular_table so they match!!!
                        fine_granular_table["season_count"] = 1
                        fine_granular_table = fine_granular_table[all_poly_and_SEOS.columns]

                        aaa = fine_granular_table.iloc[0].values.reshape(1, len(fine_granular_table.iloc[0]))

                        all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                        pointer_SEOS_tab += 1
                else: # here are potentially apples, cherries, etc.
                    # we did not add EVI_ratio, SOS, and EOS. So, we are missing these
                    # columns in the data frame. So, use 666 as proxy
                    aaa = np.append(curr_field.iloc[0][:3], [666, 666, 666, 1])
                    aaa = aaa.reshape(1, len(aaa))
                    all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                    pointer_SEOS_tab += 1

            just_season_counts = all_poly_and_SEOS[["ID", "season_count"]].copy()
            just_season_counts.drop_duplicates(inplace=True)
            just_season_counts.reset_index(drop=True, inplace=True)
            colName = VI_idx + "_" + smooth_type + "_season_count" + str(threshold)
            test_season_preds[colName] = just_season_counts["season_count"]
            
test_season_preds = pd.merge(test_season_preds, ground_truth_labels, on=["ID"], how="left")
test_season_preds.head(2)

# %%
print (list(test_season_preds.columns))
print ()
print (list(test_season_preds.columns[1:len(list(test_season_preds.columns))-8]))

# %% [markdown]
# ### Convert everything to 2 if number of seasons is more than 2

# %%
predict_cols = test_season_preds.columns[1:len(list(test_season_preds.columns))-8]
for a_col in predict_cols:
    test_season_preds.loc[test_season_preds[a_col]>2, a_col]=2

# %%
# pd_TS=fine_granular_table.copy() 
# min_season_length=40

# pd_TS_DoYDiff = pd_TS.copy()

# # find indexes of SOS and EOS
# SOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff['SOS'] != 0].tolist()
# EOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff['EOS'] != 0].tolist()

# fig, ax1 = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
#                    gridspec_kw={'hspace': 0.35, 'wspace': .05});
# ax1.grid(True)

# fine_granular_table.sort_values(by=['human_system_start_time'], inplace=True)


# curr_year=fine_granular_table.human_system_start_time.dt.year.unique()[0]

# ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table["EVI"],
#          linewidth=4, color="dodgerblue", label="smoothed");

# ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table["EVI_ratio"],
#          linewidth=2, color="gray", label="EVI_ratio");

# SOS = fine_granular_table[fine_granular_table['SOS'] != 0]
# # if len(SOS)>0: # dataframe might be empty
# #     if SOS.iloc[0]['SOS'] != 666:
# #         ax.scatter(SOS['human_system_start_time'], SOS['SOS'], marker='+', s=155, c='g', 
# #                   label="")
# #         # annotate SOS
# #         for ii in np.arange(0, len(SOS)):
# #             style = dict(size=10, color='g', rotation='vertical')
# #             ax.text(x = SOS.iloc[ii]['human_system_start_time'].date(), 
# #                     y = -0.18, 
# #                     s = str(SOS.iloc[ii]['human_system_start_time'].date())[5:], #
# #                     **style)
# #     else:
# #          ax.plot(curr_field_yr['human_system_start_time'], 
# #                  np.ones(len(curr_field_yr['human_system_start_time']))*1, 
# #                  c='g', linewidth=2);
            
# ax1.scatter(SOS['human_system_start_time'], 
#             SOS["SOS"], 
#             s=40, c="r", label="raw")

# ax1.legend(loc="upper right");

# %%

# %%
coll = "EVI_SG_season_count5"

true_single_predicted_single=0
true_single_predicted_double=0
true_double_predicted_single=0
true_double_predicted_double=0

for index_ in test_season_preds.index:
    curr_vote = list(test_season_preds[test_season_preds.index==index_].Vote)[0]
    curr_predict = list(test_season_preds.loc[test_season_preds.index==index_, coll])[0]
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
            
EVI_SG_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                      index=range(2))
EVI_SG_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
EVI_SG_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
EVI_SG_confus_tbl_test['Predict_Single']=0
EVI_SG_confus_tbl_test['Predict_Double']=0


EVI_SG_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
EVI_SG_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
EVI_SG_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
EVI_SG_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
EVI_SG_confus_tbl_test

# %%
confusion_matrix(test_season_preds["Vote"].astype(int), test_season_preds[coll].astype(int))

# %%
coll = "NDVI_SG_season_count5"

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in test_season_preds.index:
    curr_vote = list(test_season_preds[test_season_preds.index==index_].Vote)[0]
    curr_predict = list(test_season_preds.loc[test_season_preds.index==index_, coll])[0]
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
            
NDVI_SG_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                      index=range(2))
NDVI_SG_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
NDVI_SG_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
NDVI_SG_confus_tbl_test['Predict_Single']=0
NDVI_SG_confus_tbl_test['Predict_Double']=0

NDVI_SG_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
NDVI_SG_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
NDVI_SG_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
NDVI_SG_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
NDVI_SG_confus_tbl_test

# %%
coll = "EVI_regular_season_count5"

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in test_season_preds.index:
    curr_vote = list(test_season_preds[test_season_preds.index==index_].Vote)[0]
    curr_predict = list(test_season_preds.loc[test_season_preds.index==index_, coll])[0]
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
            
EVI_regular_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                      index=range(2))
EVI_regular_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
EVI_regular_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
EVI_regular_confus_tbl_test['Predict_Single']=0
EVI_regular_confus_tbl_test['Predict_Double']=0


EVI_regular_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
EVI_regular_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
EVI_regular_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
EVI_regular_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
EVI_regular_confus_tbl_test

# %%
coll = "NDVI_regular_season_count5"

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in test_season_preds.index:
    curr_vote = list(test_season_preds[test_season_preds.index==index_].Vote)[0]
    curr_predict = list(test_season_preds.loc[test_season_preds.index==index_, coll])[0]
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
            
NDVI_regular_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                      index=range(2))
NDVI_regular_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
NDVI_regular_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
NDVI_regular_confus_tbl_test['Predict_Single']=0
NDVI_regular_confus_tbl_test['Predict_Double']=0


NDVI_regular_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
NDVI_regular_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
NDVI_regular_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
NDVI_regular_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
NDVI_regular_confus_tbl_test

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
fine_granular_table = nc.create_calendar_table(SF_year=curr_YR)
fine_granular_table = pd.merge(fine_granular_table, curr_field,
                               on=['human_system_start_time', 'doy'], how='left')

# We need to fill the NAs that are created 
# because they were not created in fine_granular_table
fine_granular_table["ID"] = curr_field["ID"].unique()[0]

# replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
# uses as indicator for missing values
fine_granular_table.fillna(value={VI_idx:-1.5}, inplace=True)

fine_granular_table = nc.fill_theGap_linearLine(a_regularized_TS = fine_granular_table, V_idx=VI_idx)

fine_granular_table = nc.addToDF_SOS_EOS_White(pd_TS = fine_granular_table, VegIdx = VI_idx, 
                                               onset_thresh = onset_cut, # onset_cut
                                               offset_thresh = offset_cut) # offset_cut


###########################################################################################
import matplotlib.dates as mdates
fig, ax1 = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
ax1.grid(True);
ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table[VI_idx], 
         linewidth=4, color="dodgerblue", label=VI_idx) 

ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table[VI_idx + "_ratio"], 
         linewidth=1, color="gray", linestyle="dashed", label=VI_idx+"_ratio") 

if len(SOS_tb)>0: # dataframe might be empty
    if SOS_tb.iloc[0]['SOS'] != 666:
        ax1.scatter(SOS_tb['human_system_start_time'], SOS_tb['SOS'], marker='+', s=155, c='g', label="")
        for ii in np.arange(0, len(SOS_tb)): # annotate SOS
            style = dict(size=10, color='g', rotation='vertical')
            ax1.text(x = SOS_tb.iloc[ii]['human_system_start_time'].date(), y = -0.1, 
                    s = str(SOS_tb.iloc[ii]['human_system_start_time'].date())[5:], #
                    **style)
    else:
         ax1.plot(curr_field_yr['human_system_start_time'], 
                  np.ones(len(curr_field_yr['human_system_start_time']))*1, c='g', linewidth=2);
#  EOS
EOS = fine_granular_table[fine_granular_table['EOS'] != 0]
if len(EOS)>0: # dataframe might be empty
    if EOS.iloc[0]['EOS'] != 666:
        ax1.scatter(EOS['human_system_start_time'], EOS['EOS'], marker='+', s=155, c='r', label="")
        for ii in np.arange(0, len(EOS)): # annotate EOS
            style = dict(size=10, color='r', rotation='vertical')
            ax1.text(x = EOS.iloc[ii]['human_system_start_time'].date(), y = -0.1, 
                    s = str(EOS.iloc[ii]['human_system_start_time'].date())[5:], #[6:]
                    **style)

ax1.set_ylabel(VI_idx) # , labelpad=20); # fontsize = label_FontSize,
ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax1.legend(loc="upper right");
# ax1.set_xlim([datetime(2017, 1, 1), datetime(2018, 1, 1)])
from matplotlib.dates import MonthLocator, DateFormatter
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%b'))


###########################################################################################

# Kill false detected seasons 
fine_granular_table=nc.Null_SOS_EOS_by_DoYDiff(pd_TS=fine_granular_table, min_season_length=40)

# extract the SOS and EOS rows 
SEOS = fine_granular_table[(fine_granular_table['SOS']!=0) | fine_granular_table['EOS']!=0]
SEOS = SEOS.copy()

# SEOS = SEOS.reset_index() # not needed really
SOS_tb = fine_granular_table[fine_granular_table['SOS'] != 0]

###########################################################################################
import matplotlib.dates as mdates
fig, ax1 = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
ax1.grid(True);
ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table[VI_idx], 
         linewidth=4, color="dodgerblue", label=VI_idx) 

ax1.plot(fine_granular_table['human_system_start_time'], fine_granular_table[VI_idx + "_ratio"], 
         linewidth=1, color="gray", linestyle="dashed", label=VI_idx+"_ratio") 

if len(SOS_tb)>0: # dataframe might be empty
    if SOS_tb.iloc[0]['SOS'] != 666:
        ax1.scatter(SOS_tb['human_system_start_time'], SOS_tb['SOS'], marker='+', s=155, c='g', label="")
        # annotate SOS
        for ii in np.arange(0, len(SOS_tb)):
            style = dict(size=10, color='g', rotation='vertical')
            ax1.text(x = SOS_tb.iloc[ii]['human_system_start_time'].date(), y = -0.1, 
                    s = str(SOS_tb.iloc[ii]['human_system_start_time'].date())[5:], #
                    **style)
    else:
         ax1.plot(curr_field_yr['human_system_start_time'], 
                  np.ones(len(curr_field_yr['human_system_start_time']))*1, c='g', linewidth=2);
#  EOS
EOS = fine_granular_table[fine_granular_table['EOS'] != 0]
if len(EOS)>0: # dataframe might be empty
    if EOS.iloc[0]['EOS'] != 666:
        ax1.scatter(EOS['human_system_start_time'], EOS['EOS'], 
                   marker='+', s=155, c='r', 
                   label="")

        # annotate EOS
        for ii in np.arange(0, len(EOS)):
            style = dict(size=10, color='r', rotation='vertical')
            ax1.text(x = EOS.iloc[ii]['human_system_start_time'].date(), 
                    y = -0.1, 
                    s = str(EOS.iloc[ii]['human_system_start_time'].date())[5:], #[6:]
                    **style)

ax1.set_ylabel(VI_idx) # , labelpad=20); # fontsize = label_FontSize,
ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax1.legend(loc="upper right");
# ax1.set_xlim([datetime(2017, 1, 1), datetime(2018, 1, 1)])
from matplotlib.dates import MonthLocator, DateFormatter
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%b'))


# %%
import numpy


# %%

# %%

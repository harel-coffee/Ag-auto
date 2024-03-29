# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %%
import numpy as np
import pandas as pd
import time, datetime
import sys, os, os.path
import scipy, scipy.signal

from datetime import date, datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, \
                            confusion_matrix, balanced_accuracy_score, \
                            classification_report
import matplotlib.pyplot as plt

# from patsy import cr
# from pprint import pprint
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %% [markdown]
# ### Directories

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
meta_dir = "/Users/hn/Documents/01_research_data/NASA/0000_parameters/"
train_TS_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"

# %% [markdown]
# ### Metadata

# %%
meta = pd.read_csv(meta_dir + "evaluation_set.csv")
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

# %% [markdown]
# ### Convert everything to 2 if number of seasons is more than 2

# %%
# predict_cols = all_season_preds.columns[1:len(list(all_season_preds.columns))-8]
# for a_col in predict_cols:
#     all_season_preds.loc[all_season_preds[a_col]>2, a_col]=2

# %% [markdown]
# ### Test Set
#
# It seems $\tau=XXX$ is winner in all cases. So, let us use that on test set and report the result!

# %%
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

print (test20.shape)
test20.head(2)

# %%
test_TS_dir_base = train_TS_dir_base

for split_ID in [1, 2, 3, 4, 5]:
# for split_ID in [1]:    
    test_name_ = "test20_split_2Bconsistent_Oct17_DL_" + str(split_ID) + ".csv"
    test20 = pd.read_csv(ML_data_folder + "train_test_DL_" + str(split_ID) + "/" + test_name_)
    test_season_preds = pd.DataFrame(data = None, 
                                    index = np.arange(len(test20.ID.unique())), 
                                    columns = ["ID"])
    test_season_preds["ID"] = test20.ID.unique()

    VI_indices = ["NDVI"]
    smooth_types = ["SG"]

    for threshold in [5]:
        onset_cut=offset_cut=(threshold/10.0)
        for VI_idx in VI_indices:
            for smooth_type in smooth_types:
                # print (threshold, VI_idx, smooth_type)
                colName = VI_idx + "_" + smooth_type + "_season_count_split" + \
                          str(split_ID) + "_thresh" + str(threshold)
                if smooth_type=="SG":
                    test_TS_dir = test_TS_dir_base + "05_SG_TS/"
                else:
                    test_TS_dir = test_TS_dir_base + "04_regularized_TS/"

                file_names = [smooth_type + "_Walla2015_"         + VI_idx + "_JFD.csv", 
                              smooth_type + "_AdamBenton2016_"    + VI_idx + "_JFD.csv", 
                              smooth_type + "_Grant2017_"         + VI_idx + "_JFD.csv", 
                              smooth_type + "_FranklinYakima2018_"+ VI_idx + "_JFD.csv"]

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
                    y_orchard = curr_field[curr_field['human_system_start_time'] >= \
                                           pd.to_datetime("05-01-"+curr_YR)]
                    
                    y_orchard = y_orchard[y_orchard['human_system_start_time'] <= \
                                          pd.to_datetime("11-01-"+curr_YR)]
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
                print (test_season_preds.shape)
                print (just_season_counts.shape)
                
                just_season_counts.sort_values(by=['ID'], inplace=True)
                test_season_preds.sort_values(by=['ID'], inplace=True)

                just_season_counts.reset_index(drop=True, inplace=True)
                test_season_preds.reset_index(drop=True, inplace=True)
                
                just_season_counts.rename(columns={"season_count": colName}, inplace=True)

                # test_season_preds[colName] = just_season_counts["season_count"]
                test_season_preds = pd.merge(test_season_preds, just_season_counts, on="ID", how="left")

    test_season_preds = pd.merge(test_season_preds, ground_truth_labels, on=["ID"], how="left")
    test_season_preds.head(2)

    ### Convert everything to 2 if number of seasons is more than 2

    predict_cols = test_season_preds.columns[1:len(list(test_season_preds.columns))-8]
    for a_col in predict_cols:
        test_season_preds.loc[test_season_preds[a_col]>2, a_col]=2
        
    dir_expo = "/Users/hn/Documents/01_research_data/NASA/03_28_2024_computerReject/NDVI_ratio_mistakes/"
    os.makedirs(dir_expo, exist_ok=True)
    
    out_name = dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv"
    test_season_preds.to_csv(out_name, index = False)

    print ("----------------------------------------------------------------------------------------")
    print (f"{split_ID = }")
    CFM = confusion_matrix(test_season_preds["Vote"].astype(int), \
                            test_season_preds[colName].astype(int))

    error_ = CFM[0,1] + CFM[1,0]
    acc = (CFM[0, 0] + CFM[1, 1]) / len(test20)
    acc = acc.round(3)
    
    pred_double = CFM[0, 1] + CFM[1, 1]
    user_ = CFM[1, 1] / pred_double
    user_ = user_.round(3)
    
    actual_double = CFM[1, 0] + CFM[1, 1]
    producer_ = CFM[1, 1] / actual_double
    producer_ = producer_.round(5)
    
    print (CFM)
    print (f"{error_ = }")
    print (f"{acc = }")
    print (f"{user_ = }")
    print (f"{producer_ = }")
    producer_
    print ("----------------------------------------------------------------------------------------")

# %%
coll = colName

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

# %% [markdown]
# # The very first split

# %%
test_TS_dir_base = train_TS_dir_base
split_ID = 0
test_name_ = "test20_split_2Bconsistent_Oct17.csv"
test20 = pd.read_csv(ML_data_folder + test_name_)
test_season_preds = pd.DataFrame(data = None, 
                                index = np.arange(len(test20.ID.unique())), 
                                columns = ["ID"])
test_season_preds["ID"] = test20.ID.unique()

VI_indices = ["NDVI"]
smooth_types = ["SG"]

for threshold in [5]:
    onset_cut=offset_cut=(threshold/10.0)
    for VI_idx in VI_indices:
        for smooth_type in smooth_types:
            # print (threshold, VI_idx, smooth_type)
            colName = VI_idx + "_" + smooth_type + "_season_count_split" + \
                      str(split_ID) + "_thresh" + str(threshold)
            if smooth_type=="SG":
                test_TS_dir = test_TS_dir_base + "05_SG_TS/"
            else:
                test_TS_dir = test_TS_dir_base + "04_regularized_TS/"

            file_names = [smooth_type + "_Walla2015_"         + VI_idx + "_JFD.csv", 
                          smooth_type + "_AdamBenton2016_"    + VI_idx + "_JFD.csv", 
                          smooth_type + "_Grant2017_"         + VI_idx + "_JFD.csv", 
                          smooth_type + "_FranklinYakima2018_"+ VI_idx + "_JFD.csv"]

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
                y_orchard = curr_field[curr_field['human_system_start_time'] >= \
                                       pd.to_datetime("05-01-"+curr_YR)]

                y_orchard = y_orchard[y_orchard['human_system_start_time'] <= \
                                      pd.to_datetime("11-01-"+curr_YR)]
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
            print (test_season_preds.shape)
            print (just_season_counts.shape)

            just_season_counts.sort_values(by=['ID'], inplace=True)
            test_season_preds.sort_values(by=['ID'], inplace=True)

            just_season_counts.reset_index(drop=True, inplace=True)
            test_season_preds.reset_index(drop=True, inplace=True)

            just_season_counts.rename(columns={"season_count": colName}, inplace=True)

            # test_season_preds[colName] = just_season_counts["season_count"]
            test_season_preds = pd.merge(test_season_preds, just_season_counts, on="ID", how="left")

test_season_preds = pd.merge(test_season_preds, ground_truth_labels, on=["ID"], how="left")
test_season_preds.head(2)

### Convert everything to 2 if number of seasons is more than 2

predict_cols = test_season_preds.columns[1:len(list(test_season_preds.columns))-8]
for a_col in predict_cols:
    test_season_preds.loc[test_season_preds[a_col]>2, a_col]=2

dir_expo = "/Users/hn/Documents/01_research_data/NASA/03_28_2024_computerReject/NDVI_ratio_mistakes/"
os.makedirs(dir_expo, exist_ok=True)

out_name = dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv"
test_season_preds.to_csv(out_name, index = False)

print ("----------------------------------------------------------------------------------------")
print (f"{split_ID = }")
CFM = confusion_matrix(test_season_preds["Vote"].astype(int), \
                        test_season_preds[colName].astype(int))

error_ = CFM[0,1] + CFM[1,0]
acc = (CFM[0, 0] + CFM[1, 1]) / len(test20)
acc = acc.round(3)

pred_double = CFM[0, 1] + CFM[1, 1]
user_ = CFM[1, 1] / pred_double
user_ = user_.round(3)

actual_double = CFM[1, 0] + CFM[1, 1]
producer_ = CFM[1, 1] / actual_double
producer_ = producer_.round(5)

print (CFM)
print (f"{error_ = }")
print (f"{acc = }")
print (f"{user_ = }")
print (f"{producer_ = }")
producer_
print ("----------------------------------------------------------------------------------------")

# %%

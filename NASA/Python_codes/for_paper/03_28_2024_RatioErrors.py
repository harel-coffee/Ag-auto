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
# Computer and Electronics in Ag rejectin: Ratio Method Errors
#
#
# The notebook ```NDVI_ratio_6_testSets``` creates the confusion matrix for test set.
# It seems it does it for one split. 
#
# Its output is exported to ```/Users/hn/Documents/01_research_data/NASA/03_28_2024_computerReject/NDVI_ratio_mistakes```.
#
# We need to plot them now.

# %%
import numpy as np
import pandas as pd
import time, datetime
import sys, os, os.path
import scipy, scipy.signal

from datetime import date, datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score,\
                            confusion_matrix, balanced_accuracy_score,\
                            classification_report
import matplotlib.pyplot as plt

# from patsy import cr
# from pprint import pprint
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
meta_dir = "/Users/hn/Documents/01_research_data/NASA/0000_parameters/"
train_TS_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"

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

# %%
dir_expo = "/Users/hn/Documents/01_research_data/NASA/03_28_2024_computerReject/NDVI_ratio_mistakes/"
os.makedirs(dir_expo, exist_ok=True)

# %%
# %%time
test_TS_dir_base = train_TS_dir_base

VI_idx = "NDVI"
smooth_type = "SG"
threshold = 5
test_TS_dir = test_TS_dir_base + "05_SG_TS/"
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
    all_TS = pd.concat([all_TS, curr_file])

all_TS = all_TS[all_TS.ID.isin(list(ground_truth_labels.ID.unique()))].copy()

all_TS.reset_index(drop=True, inplace=True)
all_TS['human_system_start_time'] = pd.to_datetime(all_TS["human_system_start_time"])

# %%
# %%time
######## Read Raw files

raw_dir = train_TS_dir_base + "/data_for_train_individual_counties/"
list_files = os.listdir(raw_dir)
list_files = [x for x in list_files if x.endswith(".csv")]
list_files = [x for x in list_files if not("Monterey" in x)]

raw_TS = pd.DataFrame()
for a_file in list_files:
    df = pd.read_csv(raw_dir + a_file)
    df = df[df[VI_idx].notna()]
    df.drop(["EVI"], axis=1, inplace=True)
    raw_TS = pd.concat([raw_TS, df])

raw_TS["ID"] = raw_TS["ID"].astype(str)
raw_TS = nc.add_human_start_time_by_system_start_time(raw_TS)
raw_TS = nc.initial_clean(df=raw_TS, column_to_be_cleaned=VI_idx)
raw_TS.drop(columns=["system_start_time"], inplace=True)

# %%
all_TS.reset_index(drop=True, inplace=True)
raw_TS.reset_index(drop=True, inplace=True)

# %%
########
for split_ID in [1, 2, 3, 4, 5]:
    
    test20 = pd.read_csv(dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv")
    
    # mistakes:
    cc = "NDVI_SG_season_count_split" + str(split_ID) + "_thresh5"
    test20 = test20[test20.Vote != test20[cc]].copy()
    test20.reset_index(drop=True, inplace=True)
    
    field_IDs = test20.ID.unique()
    test_TS = all_TS[all_TS.ID.isin(test20.ID.unique())].copy()
    test_TS_raw = raw_TS[raw_TS.ID.isin(test20.ID.unique())].copy()
    
    for an_ID in field_IDs:
        curr_test = test20[test20.ID == an_ID]

        curr_YR = int(an_ID.split("_")[-1])
        curr_smooth = test_TS[test_TS['ID']==an_ID].copy()
        curr_smooth.reset_index(drop=True, inplace=True)
        curr_smooth.sort_values(by=['human_system_start_time'], inplace=True)
        curr_smooth = curr_smooth[curr_smooth.human_system_start_time.dt.year == curr_YR]

        curr_raw = test_TS_raw[test_TS_raw['ID']==an_ID].copy()
        curr_raw.reset_index(drop=True, inplace=True)
        curr_raw.sort_values(by=['human_system_start_time'], inplace=True)
        curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_YR]
        
        ####### Detect SOS-EOS and see whatis happening:
        
#         y_orchard = curr_smooth[curr_smooth['human_system_start_time'] >= \
#                                            pd.to_datetime("05-01-"+str(curr_YR))]
                    
#         y_orchard = y_orchard[y_orchard['human_system_start_time'] <= \
#                               pd.to_datetime("11-01-"+str(curr_YR))]
#         y_orchard_range = max(y_orchard[VI_idx]) - min(y_orchard[VI_idx])
#         if y_orchard_range > 0.3:
#             #######################################################################
#             ###             find SOS and EOS, and add them to the table
#             #######################################################################
#             fine_granular_table = nc.create_calendar_table(SF_year=curr_YR)
#             fine_granular_table = pd.merge(fine_granular_table, curr_smooth, 
#                                             on=['human_system_start_time', 'doy'], how='left')

#             # We need to fill the NAs that are created 
#             # because they were not created in fine_granular_table
#             fine_granular_table["ID"] = curr_smooth["ID"].unique()[0]

#             # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
#             # uses as indicator for missing values
#             fine_granular_table.fillna(value={VI_idx:-1.5}, inplace=True)

#             fine_granular_table = nc.fill_theGap_linearLine(a_regularized_TS = fine_granular_table,\
#                                                             V_idx=VI_idx)

#             fine_granular_table = nc.addToDF_SOS_EOS_White(pd_TS = fine_granular_table, 
#                                                            VegIdx = VI_idx, 
#                                                            onset_thresh = onset_cut, # onset_cut
#                                                            offset_thresh = offset_cut) # offset_cut
#             ##  Kill false detected seasons 
#             fine_granular_table=nc.Null_SOS_EOS_by_DoYDiff(pd_TS=fine_granular_table, \
#                                                            min_season_length=40)
#             # extract the SOS and EOS rows 
#             SEOS = fine_granular_table[(fine_granular_table['SOS']!=0) | fine_granular_table['EOS']!=0]
#             SEOS = SEOS.copy()
#             # SEOS = SEOS.reset_index() # not needed really
#             SOS_tb = fine_granular_table[fine_granular_table['SOS'] != 0]
#             if len(SOS_tb) >= 2:
#                 SEOS["season_count"] = len(SOS_tb)
#                 # re-order columns of SEOS so they match!!!
#                 SEOS = SEOS[all_poly_and_SEOS.columns]
#                 all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
#                 pointer_SEOS_tab += len(SEOS)
#             else:
#                 # re-order columns of fine_granular_table so they match!!!
#                 fine_granular_table["season_count"] = 1
#                 fine_granular_table = fine_granular_table[all_poly_and_SEOS.columns]

#                 aaa = fine_granular_table.iloc[0].values.reshape(1, len(fine_granular_table.iloc[0]))

#                 all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
#                 pointer_SEOS_tab += 1
#         else: # here are potentially apples, cherries, etc.
#             # we did not add EVI_ratio, SOS, and EOS. So, we are missing these
#             # columns in the data frame. So, use 666 as proxy
#             aaa = np.append(curr_smooth.iloc[0][:3], [666, 666, 666, 1])
#             aaa = aaa.reshape(1, len(aaa))
#             all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
#             pointer_SEOS_tab += 1
#         ################
        
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 3),
                    sharex='col', sharey='row',
                    gridspec_kw={'hspace': 0.1, 'wspace': .1});
        ax.grid(True); 

        ax.plot(curr_smooth["human_system_start_time"], curr_smooth[VI_idx], c="k", linewidth=2, label="SG");
        ax.scatter(curr_raw["human_system_start_time"], curr_raw[VI_idx], s=7, c="dodgerblue", label="raw");
        ax.set_ylim([-0.3, 1.15]);
        
        V = str(list(curr_test.Vote)[0])
        P = str(list(curr_test[cc])[0])
        crp = list(curr_test.CropTyp)[0]
        plot_title = f"Vote = " + V + ", pred = " + P + ", (" + crp + ")"

        ax.set_title(plot_title);
        
        plot_path = dir_expo + "split" + str(split_ID) + "_V" + V + "_pred" + P + "/"
        os.makedirs(plot_path, exist_ok=True)
        fig_name = plot_path + an_ID +'.png'
        plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")
        plt.close('all')

# %% [markdown]
# # Original Split

# %%
########
split_ID = 0
test20 = pd.read_csv(dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv")
cc = "NDVI_SG_season_count_split" + str(split_ID) + "_thresh5"
test20 = test20[test20.Vote != test20[cc]].copy()
test20.reset_index(drop=True, inplace=True)

field_IDs = test20.ID.unique()
test_TS = all_TS[all_TS.ID.isin(test20.ID.unique())].copy()
test_TS_raw = raw_TS[raw_TS.ID.isin(test20.ID.unique())].copy()

for an_ID in field_IDs:
    curr_test = test20[test20.ID == an_ID]

    curr_YR = int(an_ID.split("_")[-1])
    curr_smooth = test_TS[test_TS['ID']==an_ID].copy()
    curr_smooth.reset_index(drop=True, inplace=True)
    curr_smooth.sort_values(by=['human_system_start_time'], inplace=True)
    curr_smooth = curr_smooth[curr_smooth.human_system_start_time.dt.year == curr_YR]

    curr_raw = test_TS_raw[test_TS_raw['ID']==an_ID].copy()
    curr_raw.reset_index(drop=True, inplace=True)
    curr_raw.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_YR]

    fig, ax = plt.subplots(1, 1, figsize=(18, 3), sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0.1, 'wspace': .1});
    ax.grid(True); 

    ax.plot(curr_smooth["human_system_start_time"], curr_smooth[VI_idx], c="k", linewidth=2, label="SG");
    ax.scatter(curr_raw["human_system_start_time"], curr_raw[VI_idx], s=7, c="dodgerblue", label="raw");
    ax.set_ylim([-0.3, 1.15]);

    V = str(list(curr_test.Vote)[0])
    P = str(list(curr_test[cc])[0])
    crp = list(curr_test.CropTyp)[0]
    plot_title = f"Vote = " + V + ", pred = " + P + ", (" + crp + ")"

    ax.set_title(plot_title);

    plot_path = dir_expo + "split" + str(split_ID) + "_V" + V + "_pred" + P + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + an_ID +'.png'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")
    plt.close('all')

# %%

# %%

# %%
######## Core
for split_ID in [1, 2, 3, 4, 5]:
    test20 = pd.read_csv(dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv")
    # mistakes:
    cc = "NDVI_SG_season_count_split" + str(split_ID) + "_thresh5"
    test20 = test20[test20.Vote != test20[cc]].copy()

    test20.reset_index(drop=True, inplace=True)
    
    field_IDs = test20.ID.unique()
    test_TS = all_TS[all_TS.ID.isin(test20.ID.unique())].copy()
    test_TS_raw = raw_TS[raw_TS.ID.isin(test20.ID.unique())].copy()
    
    for an_ID in field_IDs:
        curr_test = test20[test20.ID == an_ID]

        curr_YR = int(an_ID.split("_")[-1])
        curr_smooth = test_TS[test_TS['ID']==an_ID].copy()
        curr_smooth.reset_index(drop=True, inplace=True)
        curr_smooth.sort_values(by=['human_system_start_time'], inplace=True)
        curr_smooth = curr_smooth[curr_smooth.human_system_start_time.dt.year == curr_YR]

        curr_raw = test_TS_raw[test_TS_raw['ID']==an_ID].copy()
        curr_raw.reset_index(drop=True, inplace=True)
        curr_raw.sort_values(by=['human_system_start_time'], inplace=True)
        curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_YR]
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 3),
                    sharex='col', sharey='row',
                    gridspec_kw={'hspace': 0.1, 'wspace': .1});
        ax.grid(True);
        # Plot NDVIs
        ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw, SG_dt = curr_smooth,
                                       idx = "NDVI", ax = ax,
                                       onset_cut = 0.5, offset_cut = 0.5);
        
        V = str(list(curr_test.Vote)[0])
        P = str(list(curr_test[cc])[0])
        crp = list(curr_test.CropTyp)[0]
        plot_title = f"Vote = " + V + ", pred = " + P + ", (" + crp + ")"

        ax.set_title(plot_title);
        
        plot_path = dir_expo + "more_detailed_plots/" + "split" + str(split_ID) + "_V" + V + "_pred" + P + "/"
        os.makedirs(plot_path, exist_ok=True)
        fig_name = plot_path + an_ID +'.png'
        plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")
        plt.close('all')

# %% [markdown]
# # Original Split

# %%
######## Core
split_ID = 0
test20 = pd.read_csv(dir_expo + "NDVI_ratio_testSetPreds_split" + str(split_ID) + ".csv")
cc = "NDVI_SG_season_count_split" + str(split_ID) + "_thresh5"
test20 = test20[test20.Vote != test20[cc]].copy()
test20.reset_index(drop=True, inplace=True)

field_IDs = test20.ID.unique()
test_TS = all_TS[all_TS.ID.isin(test20.ID.unique())].copy()
test_TS_raw = raw_TS[raw_TS.ID.isin(test20.ID.unique())].copy()

for an_ID in field_IDs:
    curr_test = test20[test20.ID == an_ID]

    curr_YR = int(an_ID.split("_")[-1])
    curr_smooth = test_TS[test_TS['ID']==an_ID].copy()
    curr_smooth.reset_index(drop=True, inplace=True)
    curr_smooth.sort_values(by=['human_system_start_time'], inplace=True)
    curr_smooth = curr_smooth[curr_smooth.human_system_start_time.dt.year == curr_YR]

    curr_raw = test_TS_raw[test_TS_raw['ID']==an_ID].copy()
    curr_raw.reset_index(drop=True, inplace=True)
    curr_raw.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_YR]

    fig, ax = plt.subplots(1, 1, figsize=(18, 3),
                sharex='col', sharey='row',
                gridspec_kw={'hspace': 0.1, 'wspace': .1});
    ax.grid(True);
    # Plot NDVIs
    ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw, SG_dt = curr_smooth,
                                   idx = "NDVI", ax = ax,
                                   onset_cut = 0.5, offset_cut = 0.5);

    V = str(list(curr_test.Vote)[0])
    P = str(list(curr_test[cc])[0])
    crp = list(curr_test.CropTyp)[0]
    plot_title = f"Vote = " + V + ", pred = " + P + ", (" + crp + ")"

    ax.set_title(plot_title);

    plot_path = dir_expo + "more_detailed_plots/" + "split" + str(split_ID) + "_V" + V + "_pred" + P + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + an_ID +'.png'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")
    plt.close('all')

# %%

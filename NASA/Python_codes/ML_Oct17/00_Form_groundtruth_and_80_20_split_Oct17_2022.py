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

# %% [markdown]
# Oct 17, 2022
#
# This notebook is to create ground-truth file from "train_labels.csv".
# - Check train_labels.csv to see if Experts' set 3 is included or not: **double check if fields in set 3 are done in final set or not. Include the fields with latest labels**
# - Add "Final" set to it as well. Final set is the one done in Oct 5, 2022. This was done to include "easy" crops as well.
# - Final votes of non-experts is needed as well!
# - Add what kind of crops are in this set etc. to the paper.
#
# ____________________________________________
#
# In question 9 of final survey's experts' meeting Perry said 
#
# "we can make blanket statement that canola is not double-cropped" and also carrot seed. 
#
# So, I should label all of them as single in ground-truth? 
# (if any of them is labeled as double?). 
#
# In question 15 Kirti says Hops are not double cropped. Apply this to ground-truth?
#

# %%
import pandas as pd
import csv
import numpy as np
import os, os.path
import sys

import collections # to count frequency of elements of an array
# to move files from one directory to another
import shutil

# %%
database_dir = "/Users/hn/Documents/01_research_data/NASA/"
ML_data_dir  = database_dir + "ML_data/"
params_dir   = database_dir + "parameters/"
perry_dir    = database_dir + "Perry_and_Co/"

# %%
train_labels = pd.read_csv(ML_data_dir + "train_labels.csv")

final_experts = pd.read_csv(params_dir + "NE_final_survey/" + "post_post_meeting_votes/" + 
                            "final_experts_meeting_Oct10.csv")

# %%
print (train_labels.shape)
print (final_experts.shape)
final_experts.head(2)

# %%
final_experts.rename(columns={"final_vote": "Vote"}, inplace=True)

# %%
print (final_experts.shape)
final_experts = final_experts[final_experts['Vote'].notna()].copy()
print (final_experts.shape)

# %%
final_experts.replace(to_replace="single", value=1, inplace=True)
final_experts.replace(to_replace="single ", value=1, inplace=True)
final_experts.replace(to_replace="double", value=2, inplace=True)

# %%
train_labels.head(2)

# %%

# %% [markdown]
# # Final Non-experts

# %%
easyVI_preMeeting_Agreements_Sept15 = pd.read_csv(params_dir + "NE_final_survey/" +
                                                  "easyVI_preMeeting_Agreements_Sept15.csv")

MikeDisagrees_Oct10 = pd.read_csv(params_dir + "NE_final_survey/" + "post_post_meeting_votes/" + 
                                               "MikeDisagrees_Oct10.csv")

# %%
easyVI_preMeeting_Agreements_Sept15.head(2)

# %%
easyVI_preMeeting_Agreements_Sept15.replace(to_replace="4 agree: Single Crop", value=1, inplace=True)
easyVI_preMeeting_Agreements_Sept15.replace(to_replace="4 agree: Double Crop", value=2, inplace=True)

# %%

# %%
MikeDisagrees_Oct10.head(2)

# %%
MikeDisagrees_Oct10.final_vote.unique()

# %%
MikeDisagrees_Oct10.replace(to_replace="single", value=1, inplace=True)
MikeDisagrees_Oct10.replace(to_replace="double", value=2, inplace=True)
MikeDisagrees_Oct10.rename(columns={"final_vote": "Vote"}, inplace=True)

# %%

# %%
# Are IDs Unique?
print (len(train_labels.ID) - len(train_labels.ID.unique()))

# %% [markdown]
# # Set 3

# %%
response_set_xl = pd.ExcelFile(perry_dir + "Perry_and_Co_Responses.xlsx")
response_set_sheet_names = response_set_xl.sheet_names  # see all sheet names
response_set_sheet_names = [x for x in response_set_sheet_names if 'Set 3' in x]
response_set_sheet_names = sorted(response_set_sheet_names)
response_set_sheet_names

# %%
chosen_fields = pd.read_csv(perry_dir + "set3/" + "01_Manually_Picked_IDs.csv")

print('There are [{ques_count}] questions.'.format(ques_count=chosen_fields.shape[0]))
chosen_fields.head(2)

# %%
question_count = chosen_fields.shape[0]
question_count

# %%
response_cols = ["ID", "Set", "Form", "Question",
                "PerryV", "AndrewV", # "TimV", "KirtiV",
                "PerryD", "AndrewD", # "TimD", "KirtiD",
                "PerryC", "AndrewC", # "TimC", "KirtiC"
                ]

all_responses = pd.DataFrame(columns=response_cols, 
                             index=range(question_count))


row_number = -1
for response_sheet_name in response_set_sheet_names:
    sample_response = response_set_xl.parse(response_sheet_name)
    sample_response = sample_response.drop(columns=['Timestamp'])
    number_of_questions = (sample_response.shape[1]-1)//3
    
    for question_number in range(1, number_of_questions+1):
        # list(curr_tbl.columns)[1].split("(")[1].split(")")[0]
        row_number+=1
        
        # Pick columns corresponding to current question!
        col_start = (question_number*3)-2
        curr_tbl = sample_response.iloc[:, [0, col_start, col_start+1, col_start+2]].copy()

        #  enter the Set number
        ID = curr_tbl.columns[1].split("(")[1].split(")")[0]
        all_responses.loc[row_number, "ID"]=ID
        all_responses.loc[row_number, "Set"]=response_sheet_name.split("-")[0].split(" ")[1]
        all_responses.loc[row_number, "Form"]=response_sheet_name.split("-")[1].split(" ")[1]
        all_responses.loc[row_number, "Question"]=curr_tbl.columns[1].split(" ")[1]

        for email in sample_response["Email Address"].values:

            if "andrew" in email:
                # Andrew
                all_responses.loc[row_number, "AndrewV"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][1]
                all_responses.loc[row_number, "AndrewD"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][2]
                all_responses.loc[row_number, "AndrewC"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][3]
            elif "beale" in email:
                # Perry
                all_responses.loc[row_number, "PerryV"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][1]
                all_responses.loc[row_number, "PerryD"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][2]
                all_responses.loc[row_number, "PerryC"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][3]
            elif "kirti" in email:
                # Kirti
                all_responses.loc[row_number, "KirtiV"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][1]
                all_responses.loc[row_number, "KirtiD"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][2]
                all_responses.loc[row_number, "KirtiC"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][3]
            elif "water" in email:
                # Tim
                all_responses.loc[row_number, "TimV"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][1]
                all_responses.loc[row_number, "TimD"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][2]
                all_responses.loc[row_number, "TimC"]=curr_tbl[curr_tbl["Email Address"]==email].values[0][3]
                
all_responses.head(2)

# %%
all_responses.PerryV.unique()

# %%
all_responses.AndrewV.unique()

# %%
double_place= ['Mustard Crop', 'Either double or mustard crop']

all_responses['PerryVCorrected']=all_responses['PerryV']
idx = all_responses[all_responses.PerryVCorrected.isin(double_place)].index
all_responses.loc[idx, "PerryVCorrected"] = 'Double Crop'
all_responses.head(2)

all_responses['AndrewVCorrected']=all_responses['AndrewV']
idx = all_responses[all_responses.AndrewVCorrected.isin(double_place)].index
all_responses.loc[idx, "AndrewVCorrected"] = 'Double Crop'
all_responses.head(2)
pre_meeting_agreements_set3 = all_responses[all_responses.PerryVCorrected==
                                                all_responses.AndrewVCorrected].copy()

# create Vote column in pre_meeting_agreements_set3
pre_meeting_agreements_set3['Vote'] = 1
pre_meeting_agreements_set3.head(2)

# Change Vote to 2 for double cropped fields
double_index = pre_meeting_agreements_set3[pre_meeting_agreements_set3.PerryVCorrected=="Double Crop"].index
pre_meeting_agreements_set3.loc[double_index, 'Vote']=2

pre_meeting_agreements_set3 = pre_meeting_agreements_set3[["ID", "PerryV", "AndrewV", 
                                                           "PerryVCorrected", "AndrewVCorrected", 
                                                           "Vote"]]

# %%
print (pre_meeting_agreements_set3.shape)
pre_meeting_agreements_set3.head(2)

# %%
# There are 6 fields from Set 3 that are not in the train set already! but the rest are. WTH?
pre_meeting_agreements_set3[~(pre_meeting_agreements_set3.ID.isin(list(train_labels.ID)))]

# %% [markdown]
# # Check and concatenate

# %%
# Check
easyVI_preMeeting_Agreements_Sept15[easyVI_preMeeting_Agreements_Sept15.ID.isin(list(train_labels.ID))]

# %%
train_labels = pd.concat([train_labels, easyVI_preMeeting_Agreements_Sept15[["ID", "Vote"]]])

# %%
# Check
final_experts[final_experts.ID.isin(list(train_labels.ID))]

# %%
train_labels = pd.concat([train_labels, final_experts[["ID", "Vote"]]])

# %%
# Check
MikeDisagrees_Oct10[MikeDisagrees_Oct10.ID.isin(list(train_labels.ID))]

# %%
train_labels = pd.concat([train_labels, MikeDisagrees_Oct10[["ID", "Vote"]]])
print (train_labels.shape)
train_labels.head(2)

# %%
# Check
train_labels_IDs = list(train_labels.ID)
pre_meeting_agreements_set3=pre_meeting_agreements_set3[~(pre_meeting_agreements_set3.ID.isin(train_labels_IDs))]


# %%
train_labels = pd.concat([train_labels, pre_meeting_agreements_set3[["ID", "Vote"]]])

# %%
train_labels.shape

# %%
len(train_labels.ID.unique())

# %%
fName ="agreementsOnFinalSurveySept15_RecordedonSept20.csv"
a_dir="/Users/hn/Documents/01_research_data/NASA/parameters/NE_final_survey/analysis/"
agreementsOnFSSept15 = pd.read_csv(a_dir+fName)

# %%
agreementsOnFSSept15[agreementsOnFSSept15.ID.isin(list(train_labels.ID))]

# %%
agreementsOnFSSept15.head(2)
agreementsOnFSSept15.rename(columns={"ConsensusV": "Vote"}, inplace=True)


agreementsOnFSSept15.replace(to_replace="Single", value=1, inplace=True)
agreementsOnFSSept15.replace(to_replace="Double", value=2, inplace=True)
agreementsOnFSSept15.Vote.unique()

# %%
train_labels = pd.concat([train_labels, agreementsOnFSSept15[["ID", "Vote"]]])
train_labels.head(2)

# %%
# More than 10 Acres:
evaluation_set=pd.read_csv(params_dir+"evaluation_set.csv")
evaluation_set_big = evaluation_set[evaluation_set.ExctAcr>10]
train_labels=train_labels[train_labels.ID.isin(list(evaluation_set_big.ID))]
train_labels.shape

# %%
out_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
out_name = out_dir + "groundTruth_labels_Oct17_2022.csv"
train_labels.to_csv(out_name, index = False)

# %%
train_labels.shape

# %% [markdown]
# # Check
#
# Let us check the following:
#
# In question 9 of final survey's experts' meeting Perry said 
#
# "we can make blanket statement that canola is not double-cropped" and also carrot seed. 
#
# So, I should label all of them as single in ground-truth? 
# (if any of them is labeled as double?). 
#
# In question 15 Kirti says Hops are not double cropped. Apply this to ground-truth?
#

# %%
evaluation_set.head(2)

# %%
train_labels.head(2)

# %%
train_labels = pd.merge(train_labels, evaluation_set, on=['ID'], how='left')

# %%
train_labels.head(2)

# %%
canola_trains = train_labels[train_labels.CropTyp=="canola"]
canola_trains.Vote.unique()

# %%
carrot_seed_trains = train_labels[train_labels.CropTyp=="carrot seed"]
carrot_seed_trains.Vote.unique()

# %%
hops_trains = train_labels[train_labels.CropTyp=="hops"]
hops_trains.Vote.unique()

# %%
sugar_beet_seed_trains = train_labels[train_labels.CropTyp=="sugar beet seed"]
sugar_beet_seed_trains.Vote.unique()

# %%
train_labels.shape

# %%

# %% [markdown]
# # Split train and test

# %%
ground_truth_labels = train_labels[["ID", "Vote"]].copy()
ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reset_index()

# %% [markdown]
# ### We need to have timeseries as sample data to do the split!

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

file_names = ["regular_Walla2015_EVI_JFD.csv", "regular_AdamBenton2016_EVI_JFD.csv", 
              "regular_Grant2017_EVI_JFD.csv", "regular_FranklinYakima2018_EVI_JFD.csv"]

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
data.head(2)

# %%
ground_truth_TS = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()
len(ground_truth_TS.ID.unique())

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                 index=range(len(ground_truth_TS.ID.unique())))
ground_truth_wide["ID"] = ground_truth_TS.ID.unique()

for an_ID in ground_truth_TS.ID.unique():
    curr_df = ground_truth_TS[ground_truth_TS.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%

# %%
from sklearn.model_selection import train_test_split
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)

y_test_df.shape[0]+y_train_df.shape[0]

# %%
out_name=out_dir+"train80_split_2Bconsistent_Oct17.csv"
y_train_df.to_csv(out_name, index = False)

out_name=out_dir+"test20_split_2Bconsistent_Oct17.csv"
y_test_df.to_csv(out_name, index = False)

# %%

# %%

# %%

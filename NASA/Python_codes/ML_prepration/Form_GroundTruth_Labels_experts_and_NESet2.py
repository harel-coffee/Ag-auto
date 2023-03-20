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

# %% [markdown]
# May, 23, 2022
#
# This is an update of "Form_GroundTruth_Labels_experts" notebook.
#
# In that notebook we used files such as "NE_set2_premeeting_Consensus_IDs_Votes_IncludesUnsure.csv".
# However, we now have put them together and updated some of those fields after by talking to experts
# in the 30 minutes meeting. So, we will use a new file that contains all the fields chosen for NE_Set2.

# %%
import pandas as pd
import csv

import os, os.path
import sys

# %% [markdown]
# # All chosen fields for survey 1
# A grass hay field, "106433_WSDA_SF_2017", was missed from meeting PDF! how the hell did this happen?
# - I had dropped grass hay from disagreement table! 

# %%
perry_dir = "/Users/hn/Documents/01_research_data/NASA/Perry_and_Co/"
choices_set_1_xl = pd.ExcelFile(perry_dir + "set1_PerryandCo.xlsx")
choices_set_1_sheet_names = choices_set_1_xl.sheet_names  # see all sheet names

chosen_fields_set1 = pd.DataFrame()
for a_choice_sheet in choices_set_1_sheet_names:    
    # read a damn sheet
    a_choice_sheet = choices_set_1_xl.parse(a_choice_sheet)
    chosen_fields_set1 = pd.concat([chosen_fields_set1, a_choice_sheet])

# %%
choices_set_2_xl = pd.ExcelFile(perry_dir + "set_2_handPicked.xlsx")
choices_set_2_sheet_names = choices_set_2_xl.sheet_names  # see all sheet names

chosen_fields_set2 = pd.DataFrame()
for a_choice_sheet in choices_set_2_sheet_names:    
    # read a damn sheet
    a_choice_sheet = choices_set_2_xl.parse(a_choice_sheet)
    chosen_fields_set2 = pd.concat([chosen_fields_set2, a_choice_sheet])

# %%
print (len(chosen_fields_set2.ID))
print (len(chosen_fields_set2.ID.unique()))

# %%
# # check if there are overlap between chosen fields in set 1 and set 2
# set1_uniqueIDs = list(chosen_fields_set1.ID.unique())
# set2_uniqueIDs = list(chosen_fields_set2.ID.unique())

# A = chosen_fields_set2[chosen_fields_set2.ID.isin(set1_uniqueIDs)]
# B = chosen_fields_set1[chosen_fields_set1.ID.isin(set2_uniqueIDs)]
# A.shape

# %% [markdown]
# # Expert Set 1 - post meeting

# %%
import pandas as pd

ML_data_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
expert_set_1_meeting_Consensus = pd.read_csv(ML_data_dir + "expert_set1_postmeeting_consensus.csv")
expert_set_1_meeting_Consensus.head(2)

# %%
print (expert_set_1_meeting_Consensus.shape)
print (expert_set_1_meeting_Consensus.Vote.unique())

# %% [markdown]
# # Expert Set 1 - pre-meeting

# %%
expert_set1_premeeting_consensus = pd.read_csv(ML_data_dir + "expert_set1_premeeting_consensus.csv")
expert_set1_premeeting_consensus.head(2)

# %%
drop_cols = ['PerrysVote', 'AndrewsVote', 'TimsVote',
             "PerryVCorrected", "AndrewVCorrected", "TimsVCorrected"]
expert_set1_premeeting_consensus.drop(labels=drop_cols, axis=1, inplace=True)

# %% [markdown]
# # Concatenate set 1 pre and post meeting labels

# %%
expert_set_1_labels = pd.concat([expert_set_1_meeting_Consensus, 
                                 expert_set1_premeeting_consensus[['ID', 'Vote']]])

# we need this shit, cuz some fields were requested to be discussed or had comment
expert_set_1_labels.drop_duplicates(inplace=True) 
expert_set_1_labels.shape

# %% [markdown]
# # Read Experts' Set 2 pre-meeting consensus

# %%
expert_set2_premeeting_consensus = pd.read_csv(ML_data_dir + "expert_set2_premeeting_consensus.csv")
expert_set2_premeeting_consensus.head(2)

# %%
set1_IDs = list(expert_set_1_labels.ID.unique())

#
#    Get rid of repeated fields
#
expert_set2_premeeting_consensus=expert_set2_premeeting_consensus[~expert_set2_premeeting_consensus\
                                                                  .ID.isin(set1_IDs)]

# %%
expert_set2_premeeting_consensus.head(2)

# %% [markdown]
# # Non-Expert Labels Set-2
#
# ### Pre-Meeting Consensus

# %%
NE_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/nonExpert_set2_fields/"
f_name = "NE_set2_post_expert_meeting_IDs_Votes_IncludesNA_fromOneDrive_May23.csv"
NE_S2_postExpertMeeting=pd.read_csv(NE_dir+ f_name)

NE_S2_postExpertMeeting=NE_S2_postExpertMeeting[["ID", "HosseinV"]]
NE_S2_postExpertMeeting=NE_S2_postExpertMeeting[NE_S2_postExpertMeeting.HosseinV.isin(["Single", "Double"])]

# %%
NE_S2_postExpertMeeting.head(2)

# %%
# NE_S2_PreMeeting.rename(columns = {'HosseinV':'Vote'}, inplace = True)
NE_S2_postExpertMeeting["Vote"]=0
NE_S2_postExpertMeeting.loc[NE_S2_postExpertMeeting.HosseinV=="Single", "Vote"]=1
NE_S2_postExpertMeeting.loc[NE_S2_postExpertMeeting.HosseinV=="Double", "Vote"]=2

NE_S2_postExpertMeeting=NE_S2_postExpertMeeting[["ID", "Vote"]]

# %%
expert_set2_premeeting_consensus.head(2)

# %% [markdown]
# # Concatenate DataFrames

# %%
train_labels = pd.concat([expert_set_1_labels, 
                          expert_set2_premeeting_consensus[['ID', 'Vote']],
                          NE_S2_postExpertMeeting])
train_labels.reset_index(drop=True, inplace=True)
# train_labels.drop_duplicates(inplace=True)
train_labels.shape

# %%
# out_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
# out_name = out_folder + "train_labels.csv"
# train_labels.to_csv(out_name, index = False)

# %%

# %%

# %%

# %%

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

# %%
perry_dir="/Users/hn/Documents/01_research_data/NASA/Perry_and_Co/"
param_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"

ML_data_dir="/Users/hn/Documents/01_research_data/NASA/ML_data/"

# %%
set1_all_votes_Perry=pd.read_csv(perry_dir+"set1_all_votes.csv")
print (len(set1_all_votes_Perry.ID.unique()))

# %%
evaluation_set=pd.read_csv(param_dir+"evaluation_set.csv")
print (len(evaluation_set.ID.unique()), ",", evaluation_set.shape[0])

# %%
set1_all_votes_Perry=pd.merge(set1_all_votes_Perry, 
                              evaluation_set[["ID", "CropTyp", "ExctAcr"]], 
                              on=['ID'], how='left')

# %%
######## Toss Small Fields

print (len(set1_all_votes_Perry.ID.unique()))
set1_all_votes_Perry=set1_all_votes_Perry[set1_all_votes_Perry.ExctAcr>10]
print (len(set1_all_votes_Perry.ID.unique()))

# %% [markdown]
# # Hard Crops for Experts

# %%
print (len(sorted(set1_all_votes_Perry.CropTyp.unique())))
(sorted(set1_all_votes_Perry.CropTyp.unique()))

# %% [markdown]
# # Non-Expert, Survey 2

# %%
NE_S2_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/nonExpert_set2_fields/"
nonExpert_survey2_fields=pd.read_csv(NE_S2_dir+"nonExpert_survey2_fields.csv")

# %%
print (len(nonExpert_survey2_fields.ID.unique()))
print (nonExpert_survey2_fields.shape)
nonExpert_survey2_fields=nonExpert_survey2_fields[nonExpert_survey2_fields.ExctAcr>10]
print (nonExpert_survey2_fields.shape)

# %%
print (len(sorted(nonExpert_survey2_fields.CropTyp.unique())))
((sorted(nonExpert_survey2_fields.CropTyp.unique())))

# %%
nonExpert_survey2_fields[nonExpert_survey2_fields.ID.isin(list(set1_all_votes_Perry.ID))]

# %% [markdown]
# # Expert Set 2 dammit

# %%
set2_all_responses=pd.read_csv(perry_dir+"set2_all_responses.csv")
set2_hand_picked_extended=pd.read_csv(perry_dir+"set2_hand_picked_extended.csv")

# %%
set2_all_responses.shape

# %%
print (len(set2_hand_picked_extended.ID.unique()))
set2_hand_picked_extended.shape

# %% [markdown]
# ### Some of the handpicked fields were repetitions from Set 1. Drop them

# %%
set2_hand_picked_extended=set2_hand_picked_extended[~set2_hand_picked_extended.ID.isin(set1_all_votes_Perry.ID)]
set2_hand_picked_extended=set2_hand_picked_extended[set2_hand_picked_extended.ExctAcr>10]
set2_hand_picked_extended.shape

# %%
len(set2_hand_picked_extended.ID.unique())+\
len(nonExpert_survey2_fields.ID.unique())+\
len(set1_all_votes_Perry[set1_all_votes_Perry.ExctAcr>=10].ID.unique())

# %%
len(set1_all_votes_Perry[set1_all_votes_Perry.ExctAcr>=10].ID.unique())+\
len(set1_all_votes_Perry[set1_all_votes_Perry.ExctAcr<10].ID.unique())

# %%
experts_set1_and_2_NE_set2=pd.concat([set1_all_votes_Perry[["ID", "ExctAcr", "CropTyp"]],
                                      nonExpert_survey2_fields[["ID", "ExctAcr", "CropTyp"]],
                                      set2_hand_picked_extended[["ID", "ExctAcr", "CropTyp"]]
                                      ]
                                      )

print (experts_set1_and_2_NE_set2.shape)
print (len(experts_set1_and_2_NE_set2.ID.unique()))

# %%

# %%

# %%
train_labels=pd.read_csv(ML_data_dir+"train_labels.csv")
train_labels=pd.merge(train_labels, 
                      evaluation_set[["ID", "CropTyp", "ExctAcr"]], 
                      on=['ID'], how='left')

train_labels=train_labels[train_labels.ExctAcr>=10]

# %%
train_labels[train_labels.ID.isin(list(experts_set1_and_2_NE_set2.ID))].shape

# %%
experts_set1_and_2_NE_set2[~experts_set1_and_2_NE_set2.ID.isin(list(train_labels.ID))]

# %%
print (set2_hand_picked_extended.shape)
set2_hand_picked_extended.head(2)

# %%

# %% [markdown]
# # Count No. Disagreements in Experts Set 2. 
#    
#     - Toss small fields
#     - Toss dubplicate fields from Set 1
#     - Count Disagreements
#     
# We are doing this because we did not have a second meeting with experts and therefore, those fields are not used in training set (ground-truth).

# %%
set2_all_responases_noRepetition=pd.read_csv(perry_dir+"set2_all_responases_noRepetition.csv")
print (set2_all_responases_noRepetition.shape)

# %%
set2_all_responases_noRepetition=set2_all_responases_noRepetition[set2_all_responases_noRepetition.ExctAcr>10]
set2_all_responases_noRepetition.shape

# %%
set2_all_responases_noRepetition.head(2)
set2_all_responases_noRepetition.PerryV.unique()

# %%
double_place= ['Mustard Crop', 'Either double or mustard crop']

set2_all_responases_noRepetition['PerryVCorrected']=set2_all_responases_noRepetition['PerryV']
idx = set2_all_responases_noRepetition[set2_all_responases_noRepetition.PerryVCorrected.isin(double_place)].index
set2_all_responases_noRepetition.loc[idx, "PerryVCorrected"] = 'Double Crop'
set2_all_responases_noRepetition.head(2)

set2_all_responases_noRepetition['AndrewVCorrected']=set2_all_responases_noRepetition['AndrewV']
idx = set2_all_responases_noRepetition[set2_all_responases_noRepetition.AndrewVCorrected.isin(double_place)].index
set2_all_responases_noRepetition.loc[idx, "AndrewVCorrected"] = 'Double Crop'
set2_all_responases_noRepetition.head(2)
pre_meeting_agreements = set2_all_responases_noRepetition[set2_all_responases_noRepetition.PerryVCorrected==
                                                     set2_all_responases_noRepetition.AndrewVCorrected].copy()


# %%
pre_meeting_agreements.shape

# %%
set2_all_responases_noRepetition[~(set2_all_responases_noRepetition.PerryVCorrected==
                                                     set2_all_responases_noRepetition.AndrewVCorrected)].shape

# %%

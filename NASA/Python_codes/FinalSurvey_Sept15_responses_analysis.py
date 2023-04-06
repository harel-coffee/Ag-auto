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

# %% [markdown]
# We created a final survey to include in ground-truth. 
#
# The final survey came from disagreements of WSU-experts on the initial 6000 fields (and supposedly easy fields!)
#
#
# In the meeting Mike left. So, he voted separately.
#
# We need to redo those and ask some of the hard fields from outside experts. Here we go!
#
# First find the fields that needs to be addressed by experts. Those will be final. So, on these fields we ignroe the
# disagreement between Mike and other 3 WSU-experts.

# %%
import pandas as pd
import numpy as np

# %%
param_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"

# %% [markdown]
# # Fields for Final!! survey:

# %%
f_name="easyVI_disagreements_forFinalSurvey_Sept15.csv"
easyVI_disagreements_forFinalSurvey_Sept15=pd.read_csv(param_dir+"NE_final_survey/"+f_name)

easyVI_disagreements_forFinalSurvey_Sept15.drop(labels=["Hossein", "Supriya", "Kirti", "Mike", "Vote"],
                                                axis="columns",
                                                inplace=True
                                               )

# %%
response_set_xl = pd.ExcelFile(param_dir + "NE_final_survey/"+ "easy_fiels_finalSurvey_Sept15_responses.xlsx")
response_set_sheet_names = response_set_xl.sheet_names  # see all sheet names
response_set_sheet_names = sorted(response_set_sheet_names)

# %%
question_count=easyVI_disagreements_forFinalSurvey_Sept15.shape[0]

response_cols = ["ID", "Form", "Question", "ConsensusV", "MikeV", "Marked"]
all_responses = pd.DataFrame(columns=response_cols, 
                             index=range(question_count))
row_number = -1
for response_sheet_name in response_set_sheet_names:
    sample_response = response_set_xl.parse(response_sheet_name)
    sample_response = sample_response.drop(columns=['Timestamp'])
    number_of_questions = (sample_response.shape[1]-1)//2
    
    for col in sample_response.columns:
        # list(curr_tbl.columns)[1].split("(")[1].split(")")[0]
        if "QUESTION" in col:
            question_number = int(col.split(" : ")[0].split(" ")[1])
            row_number+=1

            # Pick columns corresponding to current question!
            col_start = question_number*2-1
            curr_tbl = sample_response.iloc[:, [0, col_start, col_start+1]].copy()

            ID = curr_tbl.columns[1].split("(")[1].split(")")[0]
            all_responses.loc[row_number, "ID"]=ID
            all_responses.loc[row_number, "Form"]=response_sheet_name.split("_")[1]
            all_responses.loc[row_number, "Question"]=curr_tbl.columns[1].split(" ")[1]

            for email in sample_response["Email Address"].values:

                if "consensus" in email:
                    all_responses.loc[row_number,"ConsensusV"]=\
                               curr_tbl[curr_tbl["Email Address"]==email].values[0][1]
                    
                    all_responses.loc[row_number,"Marked"]=\
                               curr_tbl[curr_tbl["Email Address"]==email].values[0][2]
                    
                elif "wsu" in email:
                    all_responses.loc[row_number, "MikeV"]=\
                    curr_tbl[curr_tbl["Email Address"]==email].values[0][1]                
all_responses.head(2)

# %%
all_responses.tail(5)

# %%
all_responses=pd.merge(all_responses, 
                       easyVI_disagreements_forFinalSurvey_Sept15,
                        on=['ID'], how='left')
                       

# %%
all_responses.CropTyp.unique()

# %%
# Hard Crops for experts

hard_crop_types= ["sugar beet seed", "sugar beet", "canola", "sudangrass", "timothy"]
hard_crops = all_responses[all_responses.CropTyp.isin(hard_crop_types)].copy()
hard_crops.shape

# %%
marked_questions = all_responses[all_responses.Marked=="Column 1"].copy()
marked_questions.shape

# %%
# We want hard crops and marked ones to be asked from experts, but we do not want repetitions

# %%
for_experts = pd.concat([hard_crops, marked_questions])
print (for_experts.shape)
for_experts.drop_duplicates(inplace=True)
print (for_experts.shape)

# %%
for_experts.drop(labels=["Form", "Question"], axis="columns", inplace=True)
for_experts.head(2)

# %%

# %% [markdown]
# # Find where Mike Disagrees with the Rest

# %%
MikeDisagreement=all_responses[all_responses.Form.isin(["F5", "F6", "F7", "F8", "F9"])].copy()

MikeDisagreement=MikeDisagreement[MikeDisagreement.ConsensusV!=MikeDisagreement.MikeV]
print (MikeDisagreement.shape)

# Forget about those that we want to ask experts
MikeDisagreement=MikeDisagreement[~MikeDisagreement.ID.isin(list(for_experts.ID))]
MikeDisagreement.shape

# %%
out_name = param_dir+ "NE_final_survey/analysis/" + "forExpertsFinal_Sept20.csv"
for_experts.to_csv(out_name, index = False)

# MikeDisagreement.drop(labels=["Form", "Question"], axis="columns", inplace=True)
out_name = param_dir+ "NE_final_survey/analysis/" + "MikeDisagrees_Sept20.csv"
MikeDisagreement.to_csv(out_name, index = False)

# %% [markdown]
# # Agreements and not-marked

# %%
agreements = all_responses[~all_responses.ID.isin(MikeDisagreement.ID)].copy()
agreements = agreements[~agreements.ID.isin(for_experts.ID)].copy()

# agreements.drop(labels=["Form", "Question"], axis="columns", inplace=True)

out_name = param_dir+ "NE_final_survey/analysis/" + "agreementsOnFinalSurveySept15_RecordedonSept20.csv"
agreements.to_csv(out_name, index = False)

# %% [markdown]
# # Copy plots to a new directory to create google doc for going thourgh, AGAIN!

# %%
import os, os.path
import sys

# to move files from one directory to another
import shutil

# %%
VI_TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/06_SOS_plots/exactEval6000_vertLine_flat/"
out_dir_base="/Users/hn/Documents/01_research_data/NASA/final_survey_potMeetingPlots_for_anotherMeeting/"

# %%
mike_out = out_dir_base + "Michael_Consensus/"
for a_row in MikeDisagreement.index:
    plant = MikeDisagreement.loc[a_row, 'CropTyp'].lower().replace(" ", "_").replace(",", "").replace("/", "_")
    # sub_dir = VI_TS_dir + plant + "/"
    
    county = MikeDisagreement.loc[a_row, 'county']
    curr_ID = MikeDisagreement.loc[a_row, 'ID']
    
    if county == "Grant":
        county_name = "Grant2017"
    elif county in ["Adams", "Benton"]:
        county_name = "AdamBenton2016"
    elif county in ["Franklin", "Yakima"]:
        county_name = "FranklinYakima2018"
    elif county == "Walla Walla":
        county_name = "Walla2015"

    curr_filename = county_name + "_" + curr_ID + ".png"    
    output_dir = mike_out + plant + "/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        shutil.copy(VI_TS_dir + curr_filename, output_dir + curr_filename)
    except:
        print ("a_row", a_row)

# %%
MikeDisagreement.shape

# %%
''/Users/hn/Documents/01_research_data/NASA/VI_TS/06_SOS_plots/exactEval6000_vertLine_flat/FranklinYakima2018_105033_WSDA_SF_2018.png

# %%
perry_out = out_dir_base + "Perry/"
for a_row in for_experts.index:
    plant = for_experts.loc[a_row, 'CropTyp'].lower().replace(" ", "_").replace(",", "").replace("/", "_")
    # sub_dir = VI_TS_dir + plant + "/"
    
    county = for_experts.loc[a_row, 'county']
    curr_ID = for_experts.loc[a_row, 'ID']
    
    if county == "Grant":
        county_name = "Grant2017"
    elif county in ["Adams", "Benton"]:
        county_name = "AdamBenton2016"
    elif county in ["Franklin", "Yakima"]:
        county_name = "FranklinYakima2018"
    elif county == "Walla Walla":
        county_name = "Walla2015"
    
    curr_filename = county_name + "_" + curr_ID + ".png"    
    output_dir = perry_out + plant + "/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        shutil.copy(VI_TS_dir + curr_filename, output_dir + curr_filename)
    except:
        print ("a_row", a_row)

# %%
for_experts.groupby(['CropTyp'])['CropTyp'].count()

# %%
MikeDisagreement.sort_values(by=["CropTyp"])

# %%

# %%
A=all_responses[all_responses.Form.isin(["F5", "F6", "F7", "F8", "F9"])].copy()

A=A[A.ConsensusV!=A.MikeV]
print (A.shape)

# %%
for_experts

# %%

# %%
# import pandas as pd
# dir_="/Users/hn/Documents/01_research_data/NASA/parameters/NE_final_survey/"
# A = pd.read_csv(dir_ + "analysis/forExpertsFinal_Sept20.csv")

# B = pd.read_csv(dir_ + "easyVI_disagreements_forFinalSurvey_Oct10.csv")
# B=B[B.ID.isin(list(A.ID))]

# A=pd.merge(A, B[['ID', "final_vote"]],
#           on=['ID'], how='left')

# dir_="/Users/hn/Documents/01_research_data/NASA/parameters/NE_final_survey/"
# out_name = dir_ + "final_experts_meeting_Oct10.csv"
# A.to_csv(out_name, index = False)

# %%

# %%

# %%

# %%

# %%

# %%

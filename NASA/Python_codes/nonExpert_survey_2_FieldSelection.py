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
# # Second set for Non-Experts
#
# We had about 6000 fields labeled by non-experts.
# Some of them were sent to Experts!
#
# We want to do it a second time. Here,
#
#   - The fields are limited to a particular set of crops, mostly.
#   - Are chosen from the fields that are not sent to Experts.

# %%
import pandas as pd
import csv
import numpy as np
import sys, os, os.path

import collections # to count frequency of elements of an array
# to move files from one directory to another
import shutil

# %%
param_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
plots_dir = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/06_snapshot_flat_PNG/"

# %%
choices_xl = pd.ExcelFile(param_dir + "all_extended.xlsx")
choices_sheet_names = choices_xl.sheet_names  # see all sheet names

response_set_1_xl = pd.ExcelFile(param_dir + "6000responses.xlsx")
response_sheet_names = response_set_1_xl.sheet_names  # see all sheet names

print (choices_sheet_names[:5])
print (response_sheet_names[:5])

# %%
# evaluation_set_csv = pd.read_csv(param_dir + "evaluation_set.csv")
# evaluation_set_csv.drop(labels=["ExctAcr"], axis='columns', inplace=True)

# print (len(evaluation_set_csv.ID))
# print (len(evaluation_set_csv.ID.unique()))

# %%

# %% [markdown]
# # Read and assemble the choices; i.e. selected fields for survey

# %%
survey_fields = pd.DataFrame()
for a_sheet in choices_sheet_names:
    a_choice_sheet = choices_xl.parse(a_sheet)
    survey_fields = pd.concat([survey_fields, a_choice_sheet])

print (survey_fields.shape)
print (len(survey_fields.ID))
print (len(survey_fields.ID.unique()))

# %%
survey_fields.head(2)

# %% [markdown]
# ## Count number of questions

# %%
([x for x in response_sheet_names if 'problem' in x.lower()])

# %%
question_count = 0

for a_choice_sheet in choices_sheet_names:
    
    # read a damn sheet
    a_choice_sheet = choices_xl.parse(a_choice_sheet)

    # add them to the damn list
    question_count += a_choice_sheet.shape[0]

print('There are [{ques_count}] questions.'.format(ques_count=question_count))

# %% [markdown]
# # Clean Vote count start here:
#
# #### Problems:
#  - In the beginning emails were not collected! (mostly alfalfa and apple?)
#  - There are some Forms that I still do not have access to. Eshwar has not transferred them to Google Drive despite repeated emails.
#  - There are forms with repeated answers and no email.
#  - Min has responded sparsley, mostly no email, and problematic forms. 
#  - Different email for Hossein. Since I was receiving emails after completing forms, I used a fake email!! no, thank you!
#  - Problematic forms were responded by some, and after fix by others. Drop these forms.
#  - Not much Mike's vote in wheat. 

# %%
## Define the damn output dataframe
output_df = pd.DataFrame(columns=['Form', 'Question', 'ID',
                                  "Hossein", "Supriya", "Kirti", "Mike", "Min"], 
                                   index=range(question_count))
output_df.head(1)
curr_row = 0

extended_choices = pd.DataFrame()

###### populate the output datafrme

for response_sheet_name in response_sheet_names:
    # pick up the numeric part of the sheet names
    sheet_numeric_part = response_sheet_name.split()[1]
    
    # Form sheet names of choices excel sheets
    choice_sheet_name = "extended_" + sheet_numeric_part
    
    a_choice_sheet = choices_xl.parse(choice_sheet_name)
    a_response_sheet = response_set_1_xl.parse(response_sheet_name)

    # If no email is recoded, pass. We do not want it
    # a_response_sheet['Email Address'].isnull().any() works as well. 
    # It is an indication that the form did not collect emails! One or all!
    if a_response_sheet['Email Address'].isnull().all():
        continue

    
    # Fix Hossein email. (replace emails by names!)
    a_response_sheet['Email Address'] = a_response_sheet['Email Address'].str.lower()

    a_response_sheet.loc[a_response_sheet['Email Address'].str.contains('noorazar', na=False), 
                         'Email Address']="Hossein"

    a_response_sheet.loc[a_response_sheet['Email Address'].str.contains('kirti', na=False), 
                         'Email Address']="Kirti"

    a_response_sheet.loc[a_response_sheet['Email Address'].str.contains('supriya', na=False), 
                         'Email Address']="Supriya"

    a_response_sheet.loc[a_response_sheet['Email Address'].str.contains('brady', na=False), 
                         'Email Address']="Mike"

    a_response_sheet.loc[a_response_sheet['Email Address'].str.contains('ming', na=False), 
                         'Email Address']="Min"

    # Fix repeated Kirti in one sheet
    #
    #   IF Kirti responded to a problematic sheet, and then to fixed sheet,
    #   Then we have duplicates!!!!!
    #
    latest_Kirti = a_response_sheet[a_response_sheet['Email Address']=="Kirti"].Timestamp.max()
    Kirti = a_response_sheet[a_response_sheet['Email Address']=="Kirti"]
    # The fact that we have "<" below, eliminates to count how many times
    # Kirti has responded. i.e. if there is one response, it will not be dropped!
    bad_index = Kirti[Kirti.Timestamp < latest_Kirti].index
    a_response_sheet.drop(bad_index, inplace=True)

    latest_Hossein = a_response_sheet[a_response_sheet['Email Address']=="Hossein"].Timestamp.max()
    Hossein = a_response_sheet[a_response_sheet['Email Address']=="Hossein"]
    # The fact that we have "<" below, eliminates to count how many times
    # Kirti has responded. i.e. if there is one response, it will not be dropped!
    bad_index = Hossein[Hossein.Timestamp < latest_Hossein].index
    a_response_sheet.drop(bad_index, inplace=True)
    
    
    latest_Supriya = a_response_sheet[a_response_sheet['Email Address']=="Supriya"].Timestamp.max()
    Supriya = a_response_sheet[a_response_sheet['Email Address']=="Supriya"]
    # The fact that we have "<" below, eliminates to count how many times
    # Kirti has responded. i.e. if there is one response, it will not be dropped!
    bad_index = Supriya[Supriya.Timestamp < latest_Supriya].index
    a_response_sheet.drop(bad_index, inplace=True)
    
    latest_Mike = a_response_sheet[a_response_sheet['Email Address']=="Mike"].Timestamp.max()
    Mike = a_response_sheet[a_response_sheet['Email Address']=="Mike"]
    # The fact that we have "<" below, eliminates to count how many times
    # Kirti has responded. i.e. if there is one response, it will not be dropped!
    bad_index = Mike[Mike.Timestamp < latest_Mike].index
    a_response_sheet.drop(bad_index, inplace=True)
    
    latest_Min = a_response_sheet[a_response_sheet['Email Address']=="Min"].Timestamp.max()
    Min = a_response_sheet[a_response_sheet['Email Address']=="Min"]
    # The fact that we have "<" below, eliminates to count how many times
    # Kirti has responded. i.e. if there is one response, it will not be dropped!
    bad_index = Min[Min.Timestamp < latest_Min].index
    a_response_sheet.drop(bad_index, inplace=True)

    if len(a_response_sheet['Email Address']) != len(a_response_sheet['Email Address'].unique()):
        raise ValueError("Something is wrong in email address column")

    for a_col_name in a_response_sheet.columns:
        if "http" in a_col_name:
            question_number = a_col_name.split()[1].split(":")[0]
            currnt_ID = a_choice_sheet.loc[int(question_number)-1, "ID"]
            if currnt_ID in list(output_df.ID):
                curr_idx = output_df[output_df.ID == currnt_ID].index
                
                if "Hossein" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_idx, "Hossein"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Hossein"][a_col_name].values[0]

                if "Supriya" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_idx, "Supriya"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Supriya"][a_col_name].values[0]

                if "Kirti" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_idx, "Kirti"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Kirti"][a_col_name].values[0]
                if "Mike" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_idx, "Mike"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Mike"][a_col_name].values[0]
                
                if "Min" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_idx, "Min"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Min"][a_col_name].values[0]
            else:
                output_df.loc[curr_row, "ID"] = currnt_ID
                output_df.loc[curr_row, "Form"] = int(sheet_numeric_part)
                output_df.loc[curr_row, "Question"] = int(question_number)
                
                if "Hossein" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_row, "Hossein"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Hossein"][a_col_name].values[0]

                if "Supriya" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_row, "Supriya"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Supriya"][a_col_name].values[0]

                if "Kirti" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_row, "Kirti"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Kirti"][a_col_name].values[0]
                if "Mike" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_row, "Mike"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Mike"][a_col_name].values[0]
                
                if "Min" in list(a_response_sheet["Email Address"]):
                    output_df.loc[curr_row, "Min"]=a_response_sheet[a_response_sheet["Email Address"] == \
                                                                     "Min"][a_col_name].values[0]
                curr_row += 1

output_df.dropna(how='all', axis='rows', inplace=True);
print (output_df.shape)
print (len(output_df.ID))
print (len(output_df.ID.unique()))

# %%
output_df.head(2)

# %% [markdown]
# # Correct the damn responses
#
# 1. We have not been consistent in the Forms!
# 2. The options we gave there was horrible. For example, the options were:
#    - Single Crop
#    - Double Crop
#    - Unsure
#    - Other:
#    
# Convert anything that has ```cover```, ```mustard``` in it to ```double-crop```, single crops are single crops, anything else will be ```Unsure```.
#
# If an answer has any of these pairs in it, it will be labeled as unsure ```[single, double]```, ```[single, cover]```,
# ```[single, mustard]```.

# %%
output_df.Supriya.unique()

# %%
output_df.reset_index(inplace=True, drop=True)
output_backup = output_df.copy()

# %%
output_df = output_backup.copy()

# %%
output_df.Supriya.unique()

# %%
people = ["Hossein", "Supriya", "Kirti", "Mike", "Min"]

# We need to keep "none" so that we can use it to compute
# majority vote where someone has not responded a Form
output_df[people] = output_df[people].fillna('none')

for idx in output_df.index:
    for person in people:
        if output_df.loc[idx, person] in ["Single Crop", "Double Crop", "Unsure", "none"]:
            continue
            
        if "cover" in output_df.loc[idx, person].lower() or "mustard" in output_df.loc[idx, person].lower():
            output_df.loc[idx, person] = "Double Crop"
        
        else:
            output_df.loc[idx, person] = "Unsure"
            
output_df.Supriya.unique()

# %%
len(output_df.ID.unique())

# %%
output_df.Supriya.unique()

# %%
survey_fields.head(2)

# %%
extended_output = pd.merge(output_df, survey_fields, on=['ID'], how='left')
extended_output.head(2)

# %%
len(extended_output.ID.unique())

# %% [markdown]
# # Drop the fields that are labeled by Experts
#
# #### Read First set

# %%
perry_dir = "/Users/hn/Documents/01_research_data/NASA/Perry_and_Co/"
set1_fields = pd.read_csv(perry_dir + "set_1_experts_stats_extended_sortOpinionCrop.csv")
print (set1_fields.shape)
set1_fields.head(2)

# %% [markdown]
# #### Read Second Set

# %%
param_in_data_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
hand_picked_set2 = pd.read_csv(param_in_data_dir + "threeHundred_IDs_Set2_Perry.csv")
hand_picked_set2.dropna(inplace=True)
print (hand_picked_set2.shape)
hand_picked_set2.head(2)

# %% [markdown]
# #### Get unique IDs of first and second set

# %%
expert_IDs = list(hand_picked_set2.ID) + list(set1_fields.ID)
expert_IDs = list(set(expert_IDs))
len(expert_IDs)
print (len(extended_output.ID.unique()))
print (len(extended_output.ID))

# %%
extended_output.head(2)

# %% [markdown]
# ## Drop the fields that are labeled by Experts

# %%
nonExpert_extended_output = extended_output[~extended_output.ID.isin(expert_IDs)]
len(nonExpert_extended_output.ID.unique())

# %% [markdown]
# ## Keep only limited crops

# %%
wanted_crops = ['alfalfa seed',
                'barley', 'barley hay', 'bean, dry', 'bean, green', 'bluegrass seed', 'buckwheat', 
                'canola', 'carrot', 'corn seed', 'corn, field', 'corn, sweet', 
                'grass hay', 'grass seed',
                'market crops', 'mint',
                'oat hay', 'onion',
                'pea seed', 'pea, dry', 'pea, green', 'potato', 
                'triticale',
                'wheat',
                'yellow mustard']

limitCrops_nonExpert_extended = nonExpert_extended_output[nonExpert_extended_output.CropTyp.isin(wanted_crops)]
limitCrops_nonExpert_extended.shape

# %%
len(limitCrops_nonExpert_extended.ID.unique())

# %%
limitCrops_nonExpert_extended.head(2)

# %%
nonExpert_survey2_fields = survey_fields[survey_fields.ID.isin(list(limitCrops_nonExpert_extended.ID))].copy()

nonExpert_survey2_fields.sort_values(by=['CropTyp', 'ID'], inplace=True)

nonExpert_survey2_fields.reset_index(inplace=True, drop=True)
nonExpert_survey2_fields.Question_in_set = 666
nonExpert_survey2_fields.Question_overall = 666
nonExpert_survey2_fields.shape
len(nonExpert_survey2_fields.ID.unique())

# %% [markdown]
# # Write to Disc

# %%
# Be consistent with previous ones, so that the Google Scrip works
needed_cols = ["ID", 
               "NDVI_TS_Name", "corrected_RGB", "TOA_RGB", 
               "latitude", "longitude", "Question Text", "CropTyp", "Irrigtn", 
               "DataSrc", "Acres", "ExctAcr",
               "LstSrvD", "county"]

nonExpert_survey2_fields = nonExpert_survey2_fields[needed_cols]

# %%
out_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/nonExpert_set2_fields/"
out_name = out_dir + "nonExpert_survey2_fields.csv"
nonExpert_survey2_fields.to_csv(out_name, index = False)


# %%
no_questions=60

if nonExpert_survey2_fields.shape[0] % no_questions != 0:
    no_dfs=nonExpert_survey2_fields.shape[0] // no_questions + 1
else:
    no_dfs=nonExpert_survey2_fields.shape[0] // no_questions

# %%
writer_extended = pd.ExcelWriter(out_dir + 'nonExpert_survey2_fields.xlsx', engine='xlsxwriter')

for ii in range(no_dfs):
    curr_result = nonExpert_survey2_fields.loc[(ii*no_questions): ((ii+1) * no_questions) - 1, ]
    curr_result.reset_index(drop=True, inplace=True)
    curr_result.to_excel(writer_extended, sheet_name= "NE_S2_F" + str(ii+1), index=False)

writer_extended.save()

# %%
choices_xl = pd.ExcelFile(out_dir + "nonExpert_survey2_fields.xlsx")
sheet_names = choices_xl.sheet_names  # see all sheet names


survey_fields_2_check = pd.DataFrame()
for a_sheet in sheet_names:
    a_choice_sheet = choices_xl.parse(a_sheet)
    survey_fields_2_check = pd.concat([survey_fields_2_check, a_choice_sheet])

print (survey_fields_2_check.shape)
print (len(survey_fields_2_check.ID))
print (len(survey_fields_2_check.ID.unique()))

survey_fields_2_check.sort_values(by=['CropTyp', 'ID'], inplace=True)
survey_fields_2_check.reset_index(inplace=True, drop=True)

survey_fields_2_check.equals(nonExpert_survey2_fields)

# %%
len(sheet_names)

# %%
nonExpert_survey2_fields.columns

# %%

# %%

# %%

# %%

# %%

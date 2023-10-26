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
import pandas as pd
import os

import numpy as np

# %%
dir_ = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"

# %%
GT_labels = pd.read_csv(dir_ + "groundTruth_labels_Oct17_2022.csv")
GT_labels.sort_values(by=['ID'], inplace=True)
GT_labels.reset_index(inplace=True, drop=True)
GT_labels.head(2)

# %%
GT_labels[GT_labels.ID == "148385_WSDA_SF_2015"]

# %%
VI_idx = "NDVI"
smooth_type = "SG"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

file_names = [smooth_type + "_Walla2015_NDVI_JFD.csv", smooth_type + "_AdamBenton2016_NDVI_JFD.csv", 
              smooth_type + "_Grant2017_NDVI_JFD.csv", smooth_type + "_FranklinYakima2018_NDVI_JFD.csv"]

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

ground_truth_TS = data[data.ID.isin(list(GT_labels.ID.unique()))].copy()
len(ground_truth_TS.ID.unique())
del(file)

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                 index=range(len(ground_truth_TS.ID.unique())))
ground_truth_wide["ID"] = ground_truth_TS.ID.unique()

for an_ID in ground_truth_TS.ID.unique():
    curr_df = ground_truth_TS[ground_truth_TS.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "NDVI_1":"NDVI_36"] = curr_df.NDVI.values[:36]
    
ground_truth_wide.head(2)

# %%
from sklearn.model_selection import train_test_split

# %%
for state_ in [1, 2, 3, 4, 5]:
    x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                    GT_labels, 
                                                                    test_size=0.2, 
                                                                    random_state=state_,
                                                                    shuffle=True,
                                                                    stratify=GT_labels.Vote.values)
    out_dir = dir_ + "train_test_DL_" + str(state_) + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    out_name = out_dir + "train80_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    y_train_df.to_csv(out_name, index = False)

    out_name = out_dir + "test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    y_test_df.to_csv(out_name, index = False)
    
    print (f"{state_ = }")
    print (f"{y_train_df.shape = }")
    print ("------------------------------------")

# %% [markdown]
# ### Equal 1 and 2 in test set!
# How is that gonna help?

# %%
test_set_size = int(len(GT_labels) * .1)
print (test_set_size)
GT_labels_double = GT_labels[GT_labels.Vote == 2]
GT_labels_single = GT_labels[GT_labels.Vote == 1]

# %%

# %%
import random
np.random.seed(10)
doubles_test = np.random.choice(GT_labels_double.ID, size=int(test_set_size/2), replace=False)
single_test = np.random.choice(GT_labels_single.ID, size=int(test_set_size/2), replace=False)

print (f"{len(doubles_test) = }")
print (f"{len(single_test) = }")

# %%
test_set_IDs = list(doubles_test) + list(single_test)
print (f"{len(test_set_IDs) = }")
y_test_6 = GT_labels[GT_labels.ID.isin(test_set_IDs)]
y_train_6 = GT_labels[~GT_labels.ID.isin(test_set_IDs)]

# %%
state_ = 6
out_dir = dir_ + "train_test_DL_" + str(state_) + "/"
os.makedirs(out_dir, exist_ok=True)

out_name = out_dir + "train80_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
y_train_6.to_csv(out_name, index = False)

out_name = out_dir + "test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
y_test_6.to_csv(out_name, index = False)

# %% [markdown]
# ## Save wide test sets

# %%
state_ = 1

in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
f_name = in_dir + "test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
test_df = pd.read_csv(f_name)

test_df.head(2)

# %%
test_wide = ground_truth_wide[ground_truth_wide.ID.isin(list(test_df.ID))].copy()
test_wide.head(2)

# %%
test_wide = pd.merge(test_wide, GT_labels, on=["ID"], how='left')
test_wide.head(2)

# %%
for state_ in [1, 2, 3, 4, 5, 6]:
    in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
    f_name = in_dir + "test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    
    test_df = pd.read_csv(f_name)
    test_wide = ground_truth_wide[ground_truth_wide.ID.isin(list(test_df.ID))].copy()
    test_wide = pd.merge(test_wide, GT_labels, on=["ID"], how='left')
    
    out_dir = ML_data_folder + "overSamples/" + "train_test_DL_" + str(state_) + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = VI_idx +"_" + smooth_type + \
                   "_wide_test20_split_2Bconsistent_Oct17.csv"
    test_wide.to_csv(out_dir + out_file, index = False)
    
    out_dir2 = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/widen_test_TS/"
    
    out_file = VI_idx +"_" + smooth_type + \
                   "_wide_test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    test_wide.to_csv(out_dir2 + out_file, index = False)
    
    
    print (test_wide.shape)


# %%
test_wide.head(2)

# %%
GT_labels_cp = GT_labels.copy()
GT_labels_cp.rename(columns={"Vote": "label"}, inplace=True)
GT_labels_cp.head(2)

# %%
test_wide = test_wide.merge(GT_labels_cp, how='left', on='ID')
test_wide.head(2)

# %%
sum(test_wide.Vote - test_wide.label)

# %%

# %% [markdown]
# ### Copy test and train set images into different folders.
#
# Do not forget oversampling

# %%
import imblearn
from imblearn.over_sampling import RandomOverSampler
# print(imblearn.__version__)

# %% [markdown]
# ## First oversample

# %%
for state_ in range(6, 7):
    in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
    in_name = in_dir + "train80_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    y_train_df = pd.read_csv(in_name)
    print (y_train_df.shape)

# %%
A = pd.read_csv(dir_ + "overSamples/EVI_regular_wide_train80_split_2Bconsistent_Oct17_overSample3.csv")
print (A.shape)
A.head(2)

# %%
B = A.merge(GT_labels_cp, how='left', on='ID')
B.head(2)

# %%
sum(B.Vote - B.label)

# %%
state_ = 1
in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
in_name = in_dir + "train80_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
y_train_df = pd.read_csv(in_name)

# %%
y_train_df.head(2)

# %%
GT_wide = ground_truth_wide[ground_truth_wide.ID.isin(list(y_train_df.ID))].copy()
GT_wide.head(2)

# %%
y_train_df.head(2)

# %%
print (f"{GT_wide.shape=}")
print (f"{y_train_df.shape=}")

# %%
GT_wide.sort_values(by=['ID'], inplace=True)
y_train_df.sort_values(by=['ID'], inplace=True)

# %%
ratios = 0.3
oversample = RandomOverSampler(sampling_strategy=ratios, random_state=10)
X_over, y_over = oversample.fit_resample(GT_wide, y_train_df.Vote)
X_over["Vote"]=y_over

# %%
X_over.head(2)

# %%
X_over = X_over.merge(GT_labels_cp, how='left', on='ID')
X_over.head(2)

# %%
sum(X_over.Vote - X_over.label)

# %%

# %%
for state_ in range(1, 7):
    in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
    in_name = in_dir + "train80_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"
    y_train_df = pd.read_csv(in_name)
    
    # copied and modified from "OverSample.ipynb". We only need 
    # to do this for DL. So, we only need IDs. We do not need time series of NDVI.
    #
    GT_wide = ground_truth_wide[ground_truth_wide.ID.isin(list(y_train_df.ID))].copy()
    for ratios in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        oversample = RandomOverSampler(sampling_strategy=ratios, random_state=10)
        
        GT_wide.sort_values(by=['ID'], inplace=True)
        y_train_df.sort_values(by=['ID'], inplace=True)

        X_over, y_over = oversample.fit_resample(GT_wide, y_train_df.Vote)
        X_over["Vote"]=y_over

        out_dir = ML_data_folder + "overSamples/" + "train_test_DL_" + str(state_) + "/"
        os.makedirs(out_dir, exist_ok=True)
                   
        out_file = VI_idx +"_" + smooth_type + \
                   "_wide_train80_split_2Bconsistent_Oct17_overSample" + \
                   str(int(ratios*10)) + ".csv"    
        X_over.to_csv(out_dir + out_file, index = False)
        if state_== 1:
            print (X_over.shape)

# %%
print (X_over.shape)

# %% [markdown]
# # Copy 
# plots to proper locations

# %%
import sys, os, os.path, shutil

# %%
in_out_dir = dir_ + "overSamples/" + "train_test_DL_" + str(state_) + "/"
in_out_dir

# %%
state_ = 1
in_out_dir = dir_ + "overSamples/" + "train_test_DL_" + str(state_) + "/"
ratio = 0.3

f_name = VI_idx +"_" + smooth_type + \
                "_wide_train80_split_2Bconsistent_Oct17_overSample" + str(int(ratio*10)) + ".csv"    
train_df = pd.read_csv(in_out_dir + f_name)
train_df.head(2)

# %%
# %who

# %%

# %%
labeldirs = ['/single/', '/double/']

image_dir = dir_ + smooth_type + "_groundTruth_images_" + VI_idx + "/"
image_name_list = os.listdir(image_dir)
    
for state_ in range(1, 7):
    print (f"{state_ = }")

    in_out_dir = dir_ + "overSamples/" + "train_test_DL_" + str(state_) + "/"

    for ratio in [.3, .4, 0.5, .6, .7, .8]:
        # print (f"{ratio = }")
        f_name = VI_idx +"_" + smooth_type + \
                "_wide_train80_split_2Bconsistent_Oct17_overSample" + str(int(ratio*10)) + ".csv"    
        train_df = pd.read_csv(in_out_dir + f_name)
        # print (f"{train_df.shape = }")
        
        out_dir = in_out_dir + "oversample" + str(int(ratio*10)) + "/" + \
                  smooth_type + "_" + VI_idx + "_train/"
        os.makedirs(out_dir, exist_ok=True)
        # print (f"{out_dir=}")
        
        for labldir in labeldirs:
            newdir = out_dir + labldir
            os.makedirs(newdir, exist_ok=True)
            # print()
            # print(f"{newdir=}")
            
        for a_field in train_df.ID.unique():
            copy_count = (train_df.ID == a_field).sum() # faster than len(train_df[train_df.ID == a_field])

            # we need file name
            if ("single_" + a_field + ".jpg") in image_name_list:
                image_file_name = "single_" + a_field + ".jpg"
                # print (f"{image_file_name=}")
            else:
                image_file_name = "double_" + a_field + ".jpg"
            
            src = image_dir + '/' + image_file_name
            
            if image_file_name.startswith('single'):
                for copy_c in np.arange(copy_count):
                    dst = out_dir + '/single/' + \
                                  image_file_name.split(".")[0] + "_copy" + str(copy_c) + ".jpg"
                   # print ("dst: ", dst)
                    shutil.copyfile(src, dst)
            elif image_file_name.startswith('double'):
                for copy_c in np.arange(copy_count):
                    dst = out_dir + '/double/' + \
                                  image_file_name.split(".")[0] + "_copy" + str(copy_c) + ".jpg"
                    # print ("dst: ", dst)
                    shutil.copyfile(src, dst)

# %% [markdown]
# ### No need for repetition for test set.

# %%
dir_

# %%
image_dir = dir_ + smooth_type + "_groundTruth_images_" + VI_idx + "/"
image_name_list = os.listdir(image_dir)
    
for state_ in range(1, 7):
    print (f"{state_ = }")
    in_dir = dir_ + "train_test_DL_" + str(state_) + "/"
    out_dir_base = dir_ + "overSamples/" + "train_test_DL_" + str(state_) + "/"
    ###
    ###       NOTE: in DL_6 test size is 10%. However, the file name says 20%!
    ###
    f_name = "test20_split_2Bconsistent_Oct17_DL_" + str(state_) + ".csv"    
    test_df = pd.read_csv(in_dir + f_name)
    print (f"{test_df.shape = }")

    out_dir = out_dir_base + VI_idx + "_" + smooth_type + "_test/"
    os.makedirs(out_dir, exist_ok=True)
    print (f"{out_dir=}")

    for a_field in test_df.ID.unique():
        # we need file name
        if ("single_" + a_field + ".jpg") in image_name_list:
            image_file_name = "single_" + a_field + ".jpg"
        else:
            image_file_name = "double_" + a_field + ".jpg"

        src = image_dir + '/' + image_file_name
        
        if image_file_name.startswith('single'):
                dst = out_dir + image_file_name.split(".")[0] + ".jpg"
                shutil.copyfile(src, dst)
        elif image_file_name.startswith('double'):
            for copy_c in np.arange(copy_count):
                dst = out_dir + image_file_name.split(".")[0] + ".jpg"
                shutil.copyfile(src, dst)

# %%
## Test/Check stuff

# %%
dir_1 = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/train_test_DL_1/NDVI_SG_test"
dir_2 = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/train_test_DL_2/NDVI_SG_test"
dir_6 = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/train_test_DL_6/NDVI_SG_test"

# %%
dir_1_list = os.listdir(dir_1)
dir_2_list = os.listdir(dir_2)
dir_6_list = os.listdir(dir_6)

dir_1_list = [x for x in dir_1_list if x.endswith(".jpg")]
dir_2_list = [x for x in dir_2_list if x.endswith(".jpg")]
dir_6_list = [x for x in dir_6_list if x.endswith(".jpg")]

# %%
dir_1_list==dir_2_list

# %%
print (len(dir_1_list))
print (len(dir_2_list))
print (len(dir_6_list))

# %%

# %%

# %%
# for test time
test_df["VoteLetter"] = "single"
double_idx = test_df[test_df.Vote==2].index
test_df.loc[double_idx, "VoteLetter"] = "double"
test_df.head(2)

# %%
test_df.head(2)

# %%

# %%
test_df.head(2)

# %%

# %%

# %%
image_database = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/images_DL_oversample/"
overSample_subdirectory_list = sorted(next(os.walk(image_database))[1])
print (f"{image_database = }")
print ()
print (next(os.walk(image_database)))
print ()
print (next(os.walk(image_database))[1])

# %%

# %%

# %%

# %%

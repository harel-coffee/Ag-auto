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
# This was a copy of ```01_02_DL_ResultAnalysis_EVI_Regular_Exp_and_NonExp```. It included examples from experts and non-experts. Here we have merged everything and have a big dataset. Hopefully, I can do all ```EVI```, ```NDVI```, ```regular``` and ```SG```, and perhaps oversamplings here in one notebook.

# %%
import numpy as np
import pandas as pd
from datetime import date
from random import seed
from random import random
import math
import time
import scipy, scipy.signal
import os, os.path
import shutil
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow
from matplotlib.image import imread
# vgg16 model used for transfer learning on the dogs and cats dataset
from matplotlib import pyplot
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential, load_model

import tensorflow as tf
# from keras.optimizers import SGD

from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

# from keras.optimizers import gradient_descent_v2
# SGD = gradient_descent_v2.SGD(...)

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import h5py
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%

# %% [markdown]
# # Directories

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
model_dir_file_list = sorted(os.listdir(model_dir))

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
out_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/01_TL_results/"
os.makedirs(out_dir, exist_ok=True)

# %%
# print (NOAA_Ice_SST_dir_files)
print ()
print ('Number of files in model_dir is [{SR}].'.format(SR=len(model_dir_file_list)))
print ("=========================================================================================================")

TL_models_list = []
for file in model_dir_file_list:
    if ("TL" in file):
        TL_models_list+=[file]

print ('Number of TL models in model_dir is [{SR}].'.format(SR=len(TL_models_list)))

TL_models_list


# %% [markdown]
# ### Needed Functions

# %%
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224)) # load the image
    img = img_to_array(img)     # convert to array
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32') # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img


# %%
VI_idxs = ["EVI", "NDVI"]
smooth_types = ["SG", "regular"]

# overSamples_dir      = ML_data_folder + "overSamples/"
# overSample_plots_dir = ML_data_folder + "/images_DL_oversample/"
prob_thresholds = [3, 3.4, 3.5, 3.6, 4, 5, 6, 7, 8, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9]

needed_cols = ["predType_point3", "predType_point34", "predType_point35",  "predType_point36", 
               "predType_point4", 
               "predType_point5",  "predType_point6", 
               "predType_point7",  "predType_point8",
               "predType_point9",  "predType_point91", 
               "predType_point92", "predType_point93", 
               "predType_point94", "predType_point95", 
               "predType_point96", "predType_point97", 
               "predType_point98", "predType_point99"]

# %% [markdown]
# # Non-OverSample Analysis

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"

train80 = pd.read_csv(ML_data_folder + "train80_split_2Bconsistent_Oct17.csv")
test20  = pd.read_csv(ML_data_folder + "test20_split_2Bconsistent_Oct17.csv")

# training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
# ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")

ground_truth_labels = pd.concat([train80, test20])

print (test20.shape)
test20.head(2)

# %%

# %%
VI_idx = VI_idxs[0]
smooth_type = smooth_types[0]

VI_idxs = ["EVI", "NDVI"]
smooth_types = ["SG", "regular"]

for VI_idx in VI_idxs:
    for smooth_type in smooth_types:
        print (VI_idx + ", " + smooth_type)
        test_plot_dir = ML_data_folder + smooth_type + "_groundTruth_images_" + VI_idx + "/test20/"

        test_filenames = os.listdir(test_plot_dir)
        test_filenames_clean = []

        for a_file in test_filenames:
            if a_file.endswith(".jpg"):
                test_filenames_clean +=[a_file]

        # print ("len(test_filenames_clean) is [{}].".format(len(test_filenames_clean)))

        test_df = pd.DataFrame({'filename': test_filenames_clean})
        nb_samples = test_df.shape[0]

        test_df["human_predict"] = test_df.filename.str.split("_", expand=True)[0]
        test_df["prob_single"] = -1.0
        # print ("test_df.shape is {}.".format(test_df.shape))
        test_df.head(2)

        # We have done this once before. So, commented out here. and read below.
        model_name = "01_TL_" + VI_idx + "_" + smooth_type + "_train80_Oct17.h5"
        model = load_model(model_dir + model_name)
        
        for idx in test_df.index:
            img = load_image(test_plot_dir + test_df.loc[idx, 'filename'])
            test_df.loc[idx, 'prob_single'] = model.predict(img, verbose=False)[0][0]

        for prob in np.divide(prob_thresholds, 10).round(2):
            colName = "prob_point" + str(prob)[2:]
            # print ("line 39: " + str(prob))
            # print ("line 40: " + colName)
            test_df.loc[test_df.prob_single<prob, colName] = 'double'
            test_df.loc[test_df.prob_single>=prob, colName] = 'single'

        out_name = "01_" + smooth_type + "_" + VI_idx + "_TL_testPreds.csv"
        test_df.to_csv(out_dir + out_name, index = False)

print ("--------- first loop done ---------")

for VI_idx in VI_idxs:
    for smooth_type in smooth_types:
        # print ("line 3:" + VI_idx + ", " + smooth_type )
        
        out_name = "01_" + smooth_type + "_" + VI_idx + "_TL_testPreds.csv"
        test_df = pd.read_csv(out_dir + out_name)

        for ii in prob_thresholds:
            curr_prob = "prob_point" + str(ii) 
            # print ("line 10: " + curr_prob)
            curr_pred_type = "predType_point" + str(ii)
            if type(ii)==float:
                x_deci, x_inte = math.modf(ii)
                x_inte = int(x_inte)
                decimals = np.float32(x_deci)
                ss = str(int(x_inte)) + str(int((decimals*10).round()))
                curr_prob = "prob_point" + ss
                # print ("line 18: " + curr_prob)
                curr_pred_type = "predType_point" + ss

            # print ("line 12: " + curr_pred_type)
            test_df[curr_pred_type]="a"
            for idx in test_df.index:
                if test_df.loc[idx, "human_predict"]==test_df.loc[idx, curr_prob]=="single":
                    test_df.loc[idx, curr_pred_type]="True Single"
                elif test_df.loc[idx, "human_predict"]==test_df.loc[idx, curr_prob]=="double":
                    test_df.loc[idx, curr_pred_type]="True Double"
                elif test_df.loc[idx, "human_predict"]=="double" and test_df.loc[idx, curr_prob]=="single":
                    test_df.loc[idx, curr_pred_type]="False Single"
                elif test_df.loc[idx, "human_predict"]=="single" and test_df.loc[idx, curr_prob]=="double":
                    test_df.loc[idx, curr_pred_type]="False Double"
                    
        # needed_cols = ["predType_point3", "predType_point4", 
        #                "predType_point5", "predType_point6", 
        #                "predType_point7", "predType_point8",
        #                "predType_point9"]

        test_df_trimmed = test_df[needed_cols].copy()
        test_df_trimmed.head(2)

        index_rows=["True Single", "True Double", "False Double", "False Single"]
        TFR=pd.DataFrame(index=index_rows)
        for col in test_df_trimmed.columns:
            curr=pd.DataFrame(test_df_trimmed[col].value_counts())
            TFR=pd.merge(TFR, curr, left_index=True, right_index=True, how="left")
        
        TFR.fillna(0, inplace=True)
        TFR.loc["error count"] = (TFR.loc["False Double"] + TFR.loc["False Single"])
        
        out_name = "02_" + smooth_type + "_" + VI_idx + "_TL_count_TFPR.csv"
        TFR.to_csv(out_dir + out_name, index = True)

        test_df["ID"] = test_df.filename.str.split("_", expand=True)[1]+ "_" + \
                        test_df.filename.str.split("_", expand=True)[2]+ "_" + \
                        test_df.filename.str.split("_", expand=True)[3]+ "_" + \
                        test_df.filename.str.split("_", expand=True)[4].str.split(".", expand=True)[0]

        test_df = pd.merge(test_df, meta, on=['ID'], how='left')
        test_df.head(2)

        acr_predTypes = pd.DataFrame(columns=['pred_type'])
        lst = ['False Single', 'False Double', 'True Double', 'True Single']
        acr_predTypes["pred_type"] = lst

        for ii in prob_thresholds:
            curr_col = "predType_point" + str(ii)
            if type(ii)==float:
                x_deci, x_inte = math.modf(ii)
                x_inte = int(x_inte)
                decimals = np.float32(x_deci)
                curr_col = "predType_point" + str(int(x_inte)) + str(int((decimals*10).round()))
                
            # print ("line 68: " + curr_col)
            A = test_df[[curr_col, 'Acres']].groupby([curr_col]).sum()
            if type(ii)==float:
                x_deci, x_inte = math.modf(ii)
                x_inte = int(x_inte)
                decimals = np.float32(x_deci)
                ss = str(int(x_inte)) + str(int((decimals*10).round()))
                A.rename(columns={"Acres": "Acres_point"+ str(ss)}, inplace=True)
            else:
                A.rename(columns={"Acres": "Acres_point"+ str(ii)}, inplace=True)
            # print (acr_predTypes)
            acr_predTypes = pd.merge(acr_predTypes, A.reset_index(), 
                                     left_on='pred_type', right_on=curr_col,
                                     how='left')
        for ii in prob_thresholds:
            curr_col = "predType_point" + str(ii)
            if type(ii)==float:
                x_deci, x_inte = math.modf(ii)
                x_inte = int(x_inte)
                decimals = np.float32(x_deci)
                curr_col = "predType_point" + str(int(x_inte)) + str(int((decimals*10).round()))
                
            # print ("line 81: " + curr_col)
            acr_predTypes.drop(curr_col, axis="columns", inplace=True)

        sorter=index_rows
        acr_predTypes = acr_predTypes.set_index('pred_type')
        acr_predTypes.loc[sorter]

        out_name = "03_" + smooth_type + "_" + VI_idx + "_TL_Acreage_TFPR.csv"
        acr_predTypes.to_csv(out_dir + out_name, index = True)

print ("---------  cell is done ---------")

# %%

# %% [markdown]
# # OverSample Analysis

# %%
ML_data_folder       = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
overSamples_dir      = ML_data_folder + "overSamples/"
overSample_plots_dir = ML_data_folder + "/images_DL_oversample/"

train80 = pd.read_csv(ML_data_folder + "train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder  + "test20_split_2Bconsistent_Oct17.csv")

# training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
# ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")

ground_truth_labels = pd.concat([train80, test20])

print (test20.shape)
test20.head(2)

# %%
# # load an image and predict the class
# def run_example():
#     # load the image
#     VI_idx = "EVI"
#     smooth_type = "SG"
#     sample_ratio = 3
#     test_plot_dir = overSample_plots_dir + \
#                     "oversample" + str(sample_ratio) + "/" + \
#                     smooth_type + "_groundTruth_images_" + VI_idx + "/" + \
#                     "test20/"

#     img = load_image(test_plot_dir+'double_1006_WSDA_SF_2016.jpg')
    
#     # load model
#     # model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
    
#     model_name = "01_TL_" + VI_idx + "_" + smooth_type + "_train80_Oct17_oversample" + str(sample_ratio) + ".h5"
#     model = load_model(model_dir + model_name)
#     result = model.predict(img)
#     print(result[0])

# # entry point, run the example
# run_example()

# %%
sample_ratios = [3, 4, 5, 6, 7, 8]

overSample_out_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/01_TL_results/overSamples/"
os.makedirs(overSample_out_dir, exist_ok=True)
    
for VI_idx in VI_idxs:
    for smooth_type in smooth_types:
        for sample_ratio in sample_ratios:
            print (VI_idx + ", " + smooth_type + ", " + str(sample_ratio))
            
            test_plot_dir = overSample_plots_dir + \
                            "oversample" + str(sample_ratio) + "/" + \
                            smooth_type + "_groundTruth_images_" + VI_idx + "/" + \
                            "test20/"

            test_filenames = os.listdir(test_plot_dir)
            test_filenames_clean = []

            for a_file in test_filenames:
                if a_file.endswith(".jpg"):
                    test_filenames_clean +=[a_file]

            # print ("len(test_filenames_clean) is [{}].".format(len(test_filenames_clean)))

            test_df = pd.DataFrame({'filename': test_filenames_clean})
            nb_samples = test_df.shape[0]

            test_df["human_predict"] = test_df.filename.str.split("_", expand=True)[0]
            test_df["prob_single"]=-1.0
            # print ("test_df.shape is {}.".format(test_df.shape))
            test_df.head(2)

            # We have done this once before. So, commented out here. and read below.
            model_name = "01_TL_" + VI_idx + "_" + smooth_type + \
                         "_train80_Oct17_oversample" + str(sample_ratio) + ".h5"
            model = load_model(model_dir + model_name)
            for idx in test_df.index:
                img = load_image(test_plot_dir + test_df.loc[idx, 'filename'])
                test_df.loc[idx, 'prob_single'] = model.predict(img, verbose=False)[0][0]

            for prob in np.divide(prob_thresholds, 10).round(2):
                colName = "prob_point" + str(prob)[2:]
                test_df.loc[test_df.prob_single<prob, colName] = 'double'
                test_df.loc[test_df.prob_single>=prob, colName] = 'single'

            out_name = "01_" + smooth_type + "_" + VI_idx + \
                       "_TL_testPreds_SRatio_" + str(sample_ratio) + ".csv"
            test_df.to_csv(overSample_out_dir + out_name, index = False)
            
            
for VI_idx in VI_idxs:
    for smooth_type in smooth_types:
        for sample_ratio in sample_ratios:
            print (VI_idx + ", " + smooth_type + ", " + str(sample_ratio))
            out_name = "01_" + smooth_type + "_" + VI_idx + \
                       "_TL_testPreds_SRatio_" + str(sample_ratio) + ".csv"

            test_df = pd.read_csv(overSample_out_dir + out_name)

            for ii in prob_thresholds:
                curr_prob = "prob_point"+str(ii)
                curr_pred_type = "predType_point" + str(ii)
                
                if type(ii)==float:
                    x_deci, x_inte = math.modf(ii)
                    x_inte = int(x_inte)
                    decimals = np.float32(x_deci)
                    ss = str(int(x_inte)) + str(int((decimals*10).round()))
                    curr_prob = "prob_point" + ss
                    # print ("line 18: " + curr_prob)
                    curr_pred_type = "predType_point" + ss
                
                test_df[curr_pred_type]="a"
                for idx in test_df.index:
                    if test_df.loc[idx, "human_predict"]==test_df.loc[idx, curr_prob]=="single":
                        test_df.loc[idx, curr_pred_type]="True Single"
                    elif test_df.loc[idx, "human_predict"]==test_df.loc[idx, curr_prob]=="double":
                        test_df.loc[idx, curr_pred_type]="True Double"
                    elif test_df.loc[idx, "human_predict"]=="double" and test_df.loc[idx, curr_prob]=="single":
                        test_df.loc[idx, curr_pred_type]="False Single"
                    elif test_df.loc[idx, "human_predict"]=="single" and test_df.loc[idx, curr_prob]=="double":
                        test_df.loc[idx, curr_pred_type]="False Double"

#             needed_cols = ["predType_point3", "predType_point4", 
#                            "predType_point5", "predType_point6", 
#                            "predType_point7", "predType_point8",
#                            "predType_point9"]
            test_df_trimmed = test_df[needed_cols].copy()
            test_df_trimmed.head(2)

            index_rows=["True Single", "True Double", "False Double", "False Single"]
            TFR=pd.DataFrame(index=index_rows)
            for col in test_df_trimmed.columns:
                curr=pd.DataFrame(test_df_trimmed[col].value_counts())
                TFR=pd.merge(TFR, curr, left_index=True, right_index=True, how="left")
            out_name = "02_" + smooth_type + "_" + VI_idx + "_TL_count_TFPR_SRatio_" + \
                        str(sample_ratio) + ".csv"
            
            TFR.fillna(0, inplace=True)
            TFR.loc["error count"] = (TFR.loc["False Double"] + TFR.loc["False Single"])
            
            TFR.to_csv(overSample_out_dir + out_name, index = True)

            test_df["ID"] = test_df.filename.str.split("_", expand=True)[1]+ "_" + \
                            test_df.filename.str.split("_", expand=True)[2]+ "_" + \
                            test_df.filename.str.split("_", expand=True)[3]+ "_" + \
                            test_df.filename.str.split("_", expand=True)[4].str.split(".", expand=True)[0]

            test_df = pd.merge(test_df, meta, on=['ID'], how='left')
            test_df.head(2)

            acr_predTypes = pd.DataFrame(columns=['pred_type'])
            lst = ['False Single', 'False Double', 'True Double', 'True Single']
            acr_predTypes["pred_type"] = lst

            for ii in prob_thresholds:
                curr_col = "predType_point" + str(ii)                
                if type(ii)==float:
                    x_deci, x_inte = math.modf(ii)
                    x_inte = int(x_inte)
                    decimals = np.float32(x_deci)
                    ss = str(int(x_inte)) + str(int((decimals*10).round()))
                    curr_col = "prob_point" + ss
                    
                A = test_df[[curr_col, 'Acres']].groupby([curr_col]).sum()
                
                if type(ii)==float:
                    x_deci, x_inte = math.modf(ii)
                    x_inte = int(x_inte)
                    decimals = np.float32(x_deci)
                    ss = str(int(x_inte)) + str(int((decimals*10).round()))
                    A.rename(columns={"Acres": "Acres_point"+ str(ss)}, inplace=True)
                else:
                    A.rename(columns={"Acres": "Acres_point"+ str(ii)}, inplace=True)
                
                acr_predTypes = pd.merge(acr_predTypes, A.reset_index(), 
                                         left_on='pred_type', right_on=curr_col,
                                         how='left')
            for ii in prob_thresholds:
                curr_col = "predType_point" + str(ii)
                if type(ii)==float:
                    x_deci, x_inte = math.modf(ii)
                    x_inte = int(x_inte)
                    decimals = np.float32(x_deci)
                    ss = str(int(x_inte)) + str(int((decimals*10).round()))
                    curr_col = "prob_point" + ss
                    
                acr_predTypes.drop(curr_col, axis="columns", inplace=True)

            sorter=index_rows
            acr_predTypes = acr_predTypes.set_index('pred_type')
            acr_predTypes.loc[sorter]

            out_name = "03_" + smooth_type + "_" + VI_idx + "_TL_Acreage_TFPR_SRatio_" + \
                        str(sample_ratio) + ".csv"

            acr_predTypes.to_csv(overSample_out_dir + out_name, index = True)

# %%
VI_idx="EVI"
smooth_type = "SG"
out_dir = "A/" 
out_dir

# %%
import tensorflow

# %%

# %%

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
# Duplicate images for oversampling of DL

# %%
import numpy as np
import pandas as pd
from datetime import date
from random import seed
from random import random

import time
import scipy, scipy.signal
import os, os.path
import shutil
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow

# vgg16 model used for transfer learning on the dogs and cats dataset
from matplotlib import pyplot
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
# from keras.optimizers import SGD

from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# from keras.optimizers import gradient_descent_v2
# SGD = gradient_descent_v2.SGD(...)

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


import h5py
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as rcp

# %%
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# %%
# from keras.preprocessing.image import load_img # commented out in windows
# from keras.preprocessing.image import img_to_array # commented out in windows
from keras.models import load_model

# %%
overSample_database = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"
image_database = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/images_DL_oversample/"

# %%
train80_overSample7 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample7.csv")
EVI_SG_wide_train80_overSample7 = pd.read_csv(overSample_database + \
                                              "EVI_SG_wide_train80_split_2Bconsistent_Oct17_overSample7.csv")

NDVI_reg_wide_train80_overSample7 = pd.read_csv(overSample_database + \
                                              "NDVI_regular_wide_train80_split_2Bconsistent_Oct17_overSample7.csv")


# %%
print (train80_overSample7.shape)
print (EVI_SG_wide_train80_overSample7.shape)

# %%
print (list(EVI_SG_wide_train80_overSample7.ID)==list(EVI_SG_wide_train80_overSample7.ID))
print (list(EVI_SG_wide_train80_overSample7.ID)==list(NDVI_reg_wide_train80_overSample7.ID))

# %% [markdown]
# # Test sets do not depend on ratio of oversampling

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
train80.head(2)

# %%
overSample_subdirectory_list = sorted(next(os.walk(image_database))[1])
for a_sub_directory in overSample_subdirectory_list:
    VI_and_smooth_types = next(os.walk(image_database + a_sub_directory + "/"))[1]
    for a_type in VI_and_smooth_types:
        curr_directory = image_database+a_sub_directory + "/" + a_type + "/"
        test_home = curr_directory + "/test20/"
        os.makedirs(test_home, exist_ok=True)
        for file in os.listdir(curr_directory):
            if file.endswith('jpg'):
                if not ("_".join(file.split("_")[1:])[:-4] in list(train80.ID)):
                    src = curr_directory + '/' + file
                    dst = test_home + file
                    shutil.copyfile(src, dst)

# %% [markdown]
# # Copy training Sets with different oversampling ratios

# %%
overSample_subdirectory_list

# %% [markdown]
# # Prepare final dataset
#
# #### First do 80-20 split like SVM. So, everything is consistent.

# %%
train80_overSample3 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample3.csv")
train80_overSample4 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample4.csv")
train80_overSample5 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample5.csv")
train80_overSample6 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample6.csv")
train80_overSample7 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample7.csv")
train80_overSample8 = pd.read_csv(overSample_database + "train80_split_2Bconsistent_Oct17_overSample8.csv")

# %%
labeldirs = ['separate_singleDouble/single/', 'separate_singleDouble/double/']

for a_sub_directory in overSample_subdirectory_list:
    if a_sub_directory[-1]==str(3):
        train80_oversample = train80_overSample3.copy()
    elif a_sub_directory[-1]==str(4):
        train80_oversample = train80_overSample4.copy()
    elif a_sub_directory[-1]==str(5):
        train80_oversample = train80_overSample5.copy()
    elif a_sub_directory[-1]==str(6):
        train80_oversample = train80_overSample6.copy()
    elif a_sub_directory[-1]==str(7):
        train80_oversample = train80_overSample7.copy()
    elif a_sub_directory[-1]==str(8):
        train80_oversample = train80_overSample8.copy()
        
    VI_and_smooth_types = next(os.walk(image_database + a_sub_directory + "/"))[1]
    for a_type in sorted(VI_and_smooth_types):
        curr_directory = image_database + a_sub_directory + "/" + a_type + "/"
        print ("curr_directory: ", curr_directory)
        train_home = curr_directory + "/train80/"
        os.makedirs(train_home, exist_ok=True)
        
        for labldir in labeldirs:
            newdir = train_home + labldir
            os.makedirs(newdir, exist_ok=True)
            
        # We need repetition.
        for file in os.listdir(curr_directory):
            if file.endswith('jpg'):
                if ("_".join(file.split("_")[1:])[:-4] in list(train80_oversample.ID)):
                    copy_count = len(train80_oversample[train80_oversample.ID=="_".join(file.split("_")[1:])[:-4]])
                    src = curr_directory + '/' + file
                    if file.startswith('single'):
                        for copy_c in np.arange(copy_count):
                            dst = train_home + 'separate_singleDouble/single/' + \
                                  file.split(".")[0] + "_copy" + str(copy_c) + ".jpg"
                            # print ("dst: ", dst)
                            shutil.copyfile(src, dst)
                    elif file.startswith('double'):
                        for copy_c in np.arange(copy_count):
                            dst = train_home + 'separate_singleDouble/double/' + \
                                  file.split(".")[0] + "_copy" + str(copy_c) + ".jpg"
                            # print ("dst: ", dst)
                            shutil.copyfile(src, dst)

# %%
print (train80_overSample3.shape)
print (train80_overSample5.shape)
print (train80_overSample8.shape)

# %%

# %%

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
# This notebook is created on Oct 24. But the name includes Oct. 26
# to follow the same pattern as I cannot do everytihng in one day.
# It is a copy of the old notebook ```01_01_DL_model_EVI_SG.ipynb```.
#
#

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
import os

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

# %% [markdown]
# # Read Fields Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
# meta_moreThan10Acr.head(2)
# meta_moreThan10Acr_IDs = list(meta_moreThan10Acr.ID.unique())
# len(meta_moreThan10Acr_IDs)

# %%
VI_idx = "EVI"
smooth_type = "SG"

# %%

# %%
# %%time
sample_ratios = [3, 4, 5, 6, 7, 8]
for sample_ratio in sample_ratios:
    print ('Sample Ratio is {SR}.'.format(SR=sample_ratio))
    overSamples_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"
    overSample_plots_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/images_DL_oversample/"

    train_fileName = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample" + \
                     str(sample_ratio) + ".csv"
    train80 = pd.read_csv(overSamples_dir + train_fileName)

    ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
    test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")
    print (test20.shape)
    print (train80.shape)

    train_folder = overSample_plots_dir + "/oversample" + str(sample_ratio) + "/" + \
                     smooth_type + "_groundTruth_images_" + VI_idx + "/"

    print (len(os.listdir(train_folder+"train80/separate_singleDouble/single/"))+\
           len(os.listdir(train_folder+"train80/separate_singleDouble/double/")))

    train_folder_80 = train_folder +"/train80/"

    # define cnn model
    def define_model():
        # load model
        model = VGG16(include_top=False, input_shape=(224, 224, 3))
        # mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False
        # add new classifier layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # compile model
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # run the test harness for evaluating a model
    def run_test_harness():
        # define model
        _model = define_model()
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        # prepare iterator
        train_separate_dir = train_folder_80 + "/separate_singleDouble/"
        train_it = datagen.flow_from_directory(train_separate_dir,
                                               class_mode='binary', 
                                               batch_size=16, 
                                               target_size=(224, 224))
        # fit model
        _model.fit(train_it, 
                   steps_per_epoch=len(train_it), 
                   epochs=10, verbose=1)
        
        model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
        os.makedirs(model_dir, exist_ok=True)
        _model.save(model_dir+"01_TL_" + VI_idx + "_" + smooth_type + "_train80_Oct17_oversample" +\
                    str(sample_ratio) + ".h5")
        
    #     tf.keras.models.save_model(model=trained_model, filepath=model_dir+'01_TL_SingleDouble.h5')  
    #     return(_model)

    # entry point, run the test harness
    # start_time = time.time()
    run_test_harness()
    # end_time = time.time()
    

# %%

# %%
# %%time
sample_ratios = [3, 4, 5, 6, 7, 8]
for VI_idx in ["EVI", "NDVI"]:
    for smooth_type in ["SG", "regular"]:
        for sample_ratio in sample_ratios:
            print ('Sample Ratio is {SR}.'.format(SR=sample_ratio))
            overSamples_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"
            overSample_plots_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/images_DL_oversample/"

            train_fileName = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample" + \
                             str(sample_ratio) + ".csv"
            train80 = pd.read_csv(overSamples_dir + train_fileName)

            ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
            test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")
            print (test20.shape)
            print (train80.shape)

            train_folder = overSample_plots_dir + "/oversample" + str(sample_ratio) + "/" + \
                             smooth_type + "_groundTruth_images_" + VI_idx + "/"

            print (len(os.listdir(train_folder+"train80/separate_singleDouble/single/"))+\
                   len(os.listdir(train_folder+"train80/separate_singleDouble/double/")))

            train_folder_80 = train_folder +"/train80/"

            # define cnn model
            def define_model():
                # load model
                model = VGG16(include_top=False, input_shape=(224, 224, 3))
                # mark loaded layers as not trainable
                for layer in model.layers:
                    layer.trainable = False
                # add new classifier layers
                flat1 = Flatten()(model.layers[-1].output)
                class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
                output = Dense(1, activation='sigmoid')(class1)
                # define new model
                model = Model(inputs=model.inputs, outputs=output)
                # compile model
                opt = SGD(learning_rate=0.001, momentum=0.9)
                model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
                return model

            # run the test harness for evaluating a model
            def run_test_harness():
                # define model
                _model = define_model()
                # create data generator
                datagen = ImageDataGenerator(featurewise_center=True)
                # specify imagenet mean values for centering
                datagen.mean = [123.68, 116.779, 103.939]
                # prepare iterator
                train_separate_dir = train_folder_80 + "/separate_singleDouble/"
                train_it = datagen.flow_from_directory(train_separate_dir,
                                                       class_mode='binary', 
                                                       batch_size=16, 
                                                       target_size=(224, 224))
                # fit model
                _model.fit(train_it, 
                           steps_per_epoch=len(train_it), 
                           epochs=10, verbose=1)

                model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
                os.makedirs(model_dir, exist_ok=True)
                _model.save(model_dir+"01_TL_" + VI_idx + "_" + smooth_type + "_train80_Oct17_oversample" +\
                            str(sample_ratio) + ".h5")

            #     tf.keras.models.save_model(model=trained_model, filepath=model_dir+'01_TL_SingleDouble.h5')  
            #     return(_model)

            # entry point, run the test harness
            # start_time = time.time()
            run_test_harness()
            # end_time = time.time()


# %%

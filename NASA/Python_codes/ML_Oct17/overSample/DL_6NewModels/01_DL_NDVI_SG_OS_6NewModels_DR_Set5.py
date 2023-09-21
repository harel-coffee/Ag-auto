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
# This notebook is created on Oct 21, 2023.
#
# To follow the same pattern, as I cannot do everytihng in one day.
# It is a copy of the old notebook ```01_DL_model_EVI_NDVI_SG_regular_Oct17_overSamples.ipynb```.
#
#    - ```DR``` stand for Desk Reject here.
#    - ```OS``` stand for Over Sample here.

# %%
import numpy as np
import pandas as pd
from datetime import date
from random import seed, random

import time
import sys, os, os.path, shutil, h5py
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow

# vgg16 model used for transfer learning on the dogs and cats dataset
from matplotlib import pyplot

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
# from keras.optimizers import SGD

# from keras.optimizers import gradient_descent_v2
# SGD = gradient_descent_v2.SGD(...)

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as rcp

# %%
# from keras.preprocessing.image import load_img # commented out in windows
# from keras.preprocessing.image import img_to_array # commented out in windows

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
VI_idx = "NDVI"
smooth_type = "SG"

# %%

# %%

# %%
# %%time
sample_ratios = [3, 4, 5, 6, 7, 8]
train_sets = [1]    # [1, 2, 3, 4, 5, 6]

for SR in sample_ratios:
    for train_ID in train_sets:
        print (f'{SR =}, {train_ID=}')
        
        ML_data_dir_base = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
        overSamp_data_base = ML_data_dir_base + "overSamples/"
        
        train_specific_dir = overSamp_data_base + "train_test_DL_" + str(train_ID) + "/"
        
        train_plot_dir = train_specific_dir + "/oversample" + str(SR) + "/" + \
                               smooth_type + "_" + VI_idx + "_train/"

        train_fileName = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample" + \
                         str(SR) + ".csv"
        train80_wide = pd.read_csv(train_specific_dir + train_fileName)

        train_test_split_dir = ML_data_dir_base + "train_test_DL_" + str(train_ID) + "/"
        test_fName = "test20_split_2Bconsistent_Oct17_DL_"  + str(train_ID) + ".csv"
        test20 = pd.read_csv(train_test_split_dir + test_fName)
        print (test20.shape)
        print (train80_wide.shape)

        train_plot_count = len(os.listdir(train_plot_dir+"/single/"))+ \
                                    len(os.listdir(train_plot_dir+"/double/"))
        print (len(train80_wide) == train_plot_count)
        print ("===================================================")

        # train_folder_80 = train_plot_dir +"/train80/"
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
            # train_separate_dir = train_folder_80 + "/separate_singleDouble/"
            train_it = datagen.flow_from_directory(train_plot_dir,
                                                   class_mode='binary', 
                                                   batch_size=16, 
                                                   target_size=(224, 224))
            # fit model
            _model.fit(train_it, 
                       steps_per_epoch=len(train_it), 
                       epochs=10, verbose=1)

            model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/DeskReject/"
            os.makedirs(model_dir, exist_ok=True)
            _model.save(model_dir+"01_TL_" + VI_idx + "_" + smooth_type + \
                        "_train80_SR_" + str(SR) + "_DL_" +  str(train_ID) + ".h5")

        #     tf.keras.models.save_model(model=trained_model, filepath=model_dir+'01_TL_SingleDouble.h5')  
        #     return(_model)

        # entry point, run the test harness
        # start_time = time.time()
        run_test_harness()
        # end_time = time.time()


# %%

# %%

# %%

# %%

# %%


# %% [markdown]
# # There was 
#
# some code in this cell from the original notebook but I deleted to make things shorted when I do ```ctrl + F```

# %%

# %%

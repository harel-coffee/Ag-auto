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
from random import seed
from random import random

from datetime import date
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
# from keras.optimizers import SGD

from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# from keras.optimizers import gradient_descent_v2
# SGD = gradient_descent_v2.SGD(...)

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %%
idx="EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

# %%
file_names = ["SG_Walla2015_EVI_JFD.csv", "SG_AdamBenton2016_EVI_JFD.csv", 
              "SG_Grant2017_EVI_JFD.csv", "SG_FranklinYakima2018_EVI_JFD.csv"]

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
crr_fld=data[data.ID==data.ID.unique()[0]].copy()
SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]

# %%
fine_granular_table = nc.create_calendar_table(SF_year = SFYr)
fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['human_system_start_time'], how='left')

fine_granular_table.ID = crr_fld.ID.unique()[0]
# replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
# uses as indicator for missing values
fine_granular_table.fillna(value={idx:-1.5}, inplace=True)
fine_granular_table.head(2)

# %%
fine_granular_table = nc.fill_theGap_linearLine(a_regularized_TS=fine_granular_table, 
                                                V_idx=idx)

# %%
fig, ax = plt.subplots();
fig.set_size_inches(10, 2.5)
ax.grid(True);
ax.scatter(fine_granular_table['human_system_start_time'], fine_granular_table[idx], 
           marker='o', s=5, c='r', label=idx)

ax.set_xlabel('time'); # , labelpad = 15
ax.set_ylabel(idx, fontsize=12); # , labelpad = 15
ax.tick_params(axis = 'y', which = 'major');
ax.legend(loc = "upper left");

# %% [markdown]
# # Discretize

# %%
import math
epsilon = 2*math.ulp(1.0)

fine_granular_table[fine_granular_table[idx]<0][idx]=0 # Set negatives to zero
fine_granular_table[fine_granular_table[idx]==1][idx]= 1-epsilon # Avoid having 1 to avoid problem

# %%
y_bin_size = 0.01
number_of_rows = int(1/y_bin_size)
n_bins = number_of_rows

n_bins = number_of_rows

"""
  The following two lines works for EVI/NDVI, but not generally.
  It works when we are binning [0, 1]
     v = np.array([1, 2, 3, 4, 5])
     n_bins = 5
     ones_indices = np.floor(v * n_bins)
     ones_indices = list((ones_indices).astype(int))
     M = np.zeros((len(v), n_bins))
     M[np.arange(len(v)), ones_indices] = 1
"""
# ones_indices = np.floor(fine_granular_table[idx] * n_bins)
# ones_indices = list((ones_indices).astype(int))

ones_indices = list(pd.cut(x=fine_granular_table[idx], bins=np.arange(0, 1, y_bin_size), labels=False))

image_matrix = np.zeros((n_bins, len(fine_granular_table[idx])))
image_matrix[ones_indices, np.arange(len(fine_granular_table[idx]))] = 1

# %%
# M[np.arange(len(v)), ones_indices] = 1
# n_bins = number_of_rows
# v = fine_granular_table[idx]
# ones_indices = np.floor(v * n_bins)
# ones_indices = list((ones_indices).astype(int))

# M = np.zeros((len(v), n_bins))
# M[np.arange(len(v)), ones_indices] = 1

# %%
# v = np.array([.1, .2, .3, .4, .5, .9])
# n_bins = 5
# ones_indices = np.floor(v * n_bins)
# ones_indices = list((ones_indices).astype(int))
# M = np.zeros((len(v), n_bins))

# %%
from pylab import imshow
fig, ax = plt.subplots();
fig.set_size_inches(10, 2.5)
imshow(image_matrix, origin='lower')

# %%
# image_matrix3D = np.tile(image_matrix,(3, 1, 1))
image_matrix3D = np.repeat(image_matrix[:, :, np.newaxis], 3, axis=2)
image_matrix3D.shape

# %% [markdown]
# ### Plot 3D verson of the image_matrix

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

(ax1, ax2) = axs;
ax1.grid(True); ax2.grid(True)

ax1.scatter(fine_granular_table['human_system_start_time'], fine_granular_table[idx], 
            marker='o', s=5, c='r', label=idx);
left = fine_granular_table['human_system_start_time'][0]
right = fine_granular_table['human_system_start_time'].values[-1]
ax1.set_xlim([left, right]) # the following line alsow orks


ax2.imshow(image_matrix3D, origin='lower');

# plt.tight_layout()
# Make space for title
plt.subplots_adjust(top=0.85)
plt.show()

# %% [markdown]
# # Read Training Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
train_labels = pd.read_csv(training_set_dir+"train_labels.csv")
train_labels.head(2)

# %%
# # define cnn model
# def define_model():
#     # load model
#     model = VGG16(include_top=False, input_shape=(224, 224, 3))

#     # mark loaded layers as not trainable
#     for layer in model.layers:
#         layer.trainable = False
    
#     # add new classifier layers
#     flat1 = Flatten()(model.layers[-1].output)
#     class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
#     output = Dense(1, activation='sigmoid')(class1)
    
#     # define new model
#     model = Model(inputs=model.inputs, outputs=output)
    
#     # compile model
#     opt = SGD(learning_rate=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# %%
# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
train_folder = '/Users/hn/Documents/01_research_data/dogs-vs-cats/train/'
test_folder = "/Users/hn/Documents/01_research_data/dogs-vs-cats/test1/"
# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # define filename
    filename = train_folder + 'dog.' + str(i) + '.jpg'
    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()

# %%
from keras.preprocessing.image import load_img
photo = load_img(train_folder + 'dog.' + str(0) + '.jpg', target_size=(200, 200))
print (photo.size)
photo

# %% [markdown]
# # Pre-Process Photo Sizes (Optional)

# %% [raw]
# # Pre-Process Photo Sizes (Optional)
# # load dogs vs cats dataset, reshape and save to a new file
# from numpy import asarray
# from numpy import save
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# # define location of dataset
# folder = train_folder
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in os.listdir(folder):
#     # determine class
#     output = 0.0
#     if file.startswith('dog'):
#         output = 1.0
#     # load image
#     photo = load_img(folder + file, target_size=(200, 200))
#     # convert to numpy array
#     photo = img_to_array(photo)
#     # store
#     photos.append(photo)
#     labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save(folder + 'dogs_vs_cats_photos.npy', photos)
# save(folder + 'dogs_vs_cats_labels.npy', labels)

# %% [markdown]
# # Load prepared data

# %%
# # load and confirm the shape
# from numpy import load
# photos = load(train_folder +'dogs_vs_cats_photos.npy')
# labels = load(train_folder +'dogs_vs_cats_labels.npy')
# print(photos.shape, labels.shape)

# %%
print (train_folder)
len(os.listdir(train_folder))
# os.chdir('/tmp')

# %%
# create directories
dataset_home = train_folder + 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']

for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        os.makedirs(newdir, exist_ok=True)

# %%
#
# We have done this once before! So, I comment out this cell.
#
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25

# # copy training dataset images into subdirectories

src_directory = train_folder
for file in os.listdir(src_directory)[:2000]:
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        shutil.copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/'  + file
        shutil.copyfile(src, dst)

# %%

# %% [markdown]
# ### Develop a Baseline CNN Model
#
# # Skip some steps here and jump to transfer learning

# %%
train_folder="/Users/hn/Documents/01_research_data/dogs-vs-cats/train/dataset_dogs_vs_cats/train/"
test_folder="/Users/hn/Documents/01_research_data/dogs-vs-cats/train/dataset_dogs_vs_cats/test/"

# %%
train_folder


# %%
# define CNN model
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

# plot diagnostic learning curves
def summarize_diagnostics(_history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(_history.history['loss'], color='blue', label='train')
    pyplot.plot(_history.history['val_loss'], color='orange', label='test')
    pyplot.legend(loc = "upper right");
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(_history.history['accuracy'], color='blue', label='train')
    pyplot.plot(_history.history['val_accuracy'], color='orange', label='test')
    
    pyplot.subplots_adjust(left=0.1,
                           bottom=0.1, 
                           right=0.9, 
                           top=0.9, 
                           wspace=0.4, 
                           hspace=0.4)
    pyplot.legend(loc = "upper left");
    
    # save plot to file
    plot_dir = "/Users/hn/Documents/01_research_data/dogs-vs-cats/"
    filename = plot_dir + "my_plot_one.png"
    pyplot.savefig(filename, dpi=400, bbox_inches='tight')
    pyplot.close()
    
# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory(train_folder, # 'dataset_dogs_vs_cats/train/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    test_it = datagen.flow_from_directory(test_folder, # 'dataset_dogs_vs_cats/test/',
                                          class_mode='binary', batch_size=64, target_size=(224, 224))
    # fit model
    _history = model.fit(train_it, steps_per_epoch=len(train_it),
                         validation_data=test_it, 
                         validation_steps=len(test_it), 
                         epochs=5, verbose=1) # epochs=10
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0) # model.evaluate_generator
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(_history)
    return (_history)


# %%
# entry point, run the test harness
# # %time
start_time = time.time()
a_model_with_history = run_test_harness()
end_time = time.time()

# %%
summarize_diagnostics(a_model_with_history)

# %% [markdown]
# # Finalize the Model and Make Predictions
#
# **Prepare Final Dataset**
#
# A final model is typically fit on all available data, such as the combination of all train and test datasets.

# %% [markdown]
# # Prepare final dataset

# %%
# organize dataset into a useful structure
# create directories
dataset_home = '/Users/hn/Documents/01_research_data/dogs-vs-cats/train/'
# create label subdirectories
labeldirs = ['separate_dog_cat/dogs/', 'separate_dog_cat/cats/']
for labldir in labeldirs:
    newdir = dataset_home + labldir
    os.makedirs(newdir, exist_ok=True)
    
# # copy training dataset images into subdirectories
src_directory = dataset_home
for file in os.listdir(src_directory)[:2000]:  # I am doing this so that the training won't take too long
    src = src_directory + '/' + file
    if file.startswith('cat'):
        dst = dataset_home + 'separate_dog_cat/cats/'  + file
        shutil.copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + 'separate_dog_cat/dogs/'  + file
        shutil.copyfile(src, dst)


# %%
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
    train_separate_dir = "/Users/hn/Documents/01_research_data/dogs-vs-cats/train/separate_dog_cat/"
    train_it = datagen.flow_from_directory(train_separate_dir,
                                           class_mode='binary', 
                                           batch_size=64, 
                                           target_size=(224, 224))
    # fit model
    _model = _model.fit(train_it, 
                        steps_per_epoch=len(train_it), 
                        epochs=10, verbose=1)
    # save model
    model_dir = "/Users/hn/Documents/01_research_data/dogs-vs-cats/"
    _model.save(model_dir+'final_model_cats_dogs.h5')
    return(_model)

# entry point, run the test harness
start_time = time.time()
trained_model = run_test_harness()
end_time = time.time()

# %%

# %%

# %%

# %%
# plot loss
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(trained_model.history['loss'], color='blue', label='train')
# pyplot.plot(trained_model.history['val_loss'], color='orange', label='test')
pyplot.legend(loc = "upper right");

# plot accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(trained_model.history['accuracy'], color='blue', label='train')
# pyplot.plot(trained_model.history['val_accuracy'], color='orange', label='test')
pyplot.legend(loc = "upper left");

# save plot to file
plot_dir = "/Users/hn/Documents/01_research_data/dogs-vs-cats/"
filename = plot_dir + "my_plot_one.png"
pyplot.savefig(filename, dpi=300)
pyplot.close()


summarize_diagnostics(trained_model)

# %%
trained_model.history['accuracy']

# %%

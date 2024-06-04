# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Oversample results about big-ratio does not mean the best

# %%
import numpy as np
import pandas as pd
from datetime import date
from random import seed, random
import math
import time
import scipy, scipy.signal
import shutil
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow
from matplotlib.image import imread
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
import os, os.path, sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
in_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/01_TL_results/overSamples/"

# %%
in_dir_file_list = sorted(os.listdir(in_dir))

in_dir_file_list_clean = []
for file in in_dir_file_list:
    if ("02" in file):
        in_dir_file_list_clean+=[file]


# %%
SG_NDVI = []
SG_EVI  = []
regular_NDVI = []
regular_EVI  = []

for file in in_dir_file_list_clean:
    if "SG_EVI" in file:
        SG_EVI+=[file]
    elif "SG_NDVI" in file:
        SG_NDVI += [file]
    elif "regular_NDVI" in file:
        regular_NDVI += [file]
    elif "regular_EVI" in file:
        regular_EVI += [file]
    else:
        print ("fuck")

# %%
SG_NDVI

# %%
ratios_ = ["SRatio_3", "SRatio_4", "SRatio_5", "SRatio_6", "SRatio_7", "SRatio_8"]
columns_ = ["ratio", "SG_EVI", "SG_NDVI", "regular_NDVI", "regular_EVI"]

best_on_test = pd.DataFrame(columns=columns_, index=np.arange(len(ratios_)))
best_on_test.ratio = ratios_
best_on_test

# %%
for file in SG_NDVI:
    curr_ratio = file.split("_")[-1].split(".")[0]
    curr_ratio = "SRatio_" + curr_ratio
    current_file = pd.read_csv(in_dir + file, index_col=0)
    curr_min = current_file.loc["error count"].min()
    best_on_test.loc[best_on_test.ratio==curr_ratio, "SG_NDVI"]=curr_min
    
for file in SG_EVI:
    curr_ratio = file.split("_")[-1].split(".")[0]
    curr_ratio = "SRatio_" + curr_ratio
    current_file = pd.read_csv(in_dir + file, index_col=0)
    curr_min = current_file.loc["error count"].min()
    best_on_test.loc[best_on_test.ratio==curr_ratio, "SG_EVI"]=curr_min
    
for file in regular_NDVI:
    curr_ratio = file.split("_")[-1].split(".")[0]
    curr_ratio = "SRatio_" + curr_ratio
    current_file = pd.read_csv(in_dir + file, index_col=0)
    curr_min = current_file.loc["error count"].min()
    best_on_test.loc[best_on_test.ratio==curr_ratio, "regular_NDVI"]=curr_min
    
for file in regular_EVI:
    curr_ratio = file.split("_")[-1].split(".")[0]
    curr_ratio = "SRatio_" + curr_ratio
    current_file = pd.read_csv(in_dir + file, index_col=0)
    curr_min = current_file.loc["error count"].min()
    best_on_test.loc[best_on_test.ratio==curr_ratio, "regular_EVI"]=curr_min

# %%
best_on_test

# %% [markdown]
# ## Find best prob. for putting in paper

# %%
SG_EVI_bestSR = "SRatio_7"
regular_EVI_bestSR = "SRatio_4"
SG_NDVI_bestSR = "SRatio_5"
regular_NDVI_bestSR = "SRatio_5"

# %%
# SG_EVI_bestSR
current_file = pd.read_csv(in_dir + "02_SG_EVI_TL_count_TFPR_SRatio_7.csv", index_col=0)
curr_min = current_file.loc["error count"].min()
print (f"{curr_min=}")
L = list(current_file.loc["error count"])
print (f"{L.index(min(L))=}")
current_file

# %%
# regular_EVI_bestSR
current_file = pd.read_csv(in_dir + "02_regular_EVI_TL_count_TFPR_SRatio_4.csv", index_col=0)
curr_min = current_file.loc["error count"].min()
print (f"{curr_min=}")
L = list(current_file.loc["error count"])
print (f"{L.index(min(L))=}")
current_file

# %%
# regular_NDVI_bestSR
current_file = pd.read_csv(in_dir + "02_regular_NDVI_TL_count_TFPR_SRatio_5.csv", index_col=0)
curr_min = current_file.loc["error count"].min()
print (f"{curr_min=}")
L = list(current_file.loc["error count"])
print (f"{L.index(min(L))=}")
current_file

# %%
# SG_NDVI_bestSR
current_file = pd.read_csv(in_dir + "02_SG_NDVI_TL_count_TFPR_SRatio_5.csv", index_col=0)
curr_min = current_file.loc["error count"].min()
print (f"{curr_min=}")
L = list(current_file.loc["error count"])
print (f"{L.index(min(L))=}")
current_file

# %% [markdown]
# # Form Confusion Tables
#
# Minimum errors are given in ```best_on_test```. We can just look into those files
# and see how many columns are there with the same error rate. Pick the one that maximizes ```True Double```!

# %%
labels_ = ["True Single", "True Double", "False Double", "False Single"]
columns_ = ["label", "SG_EVI", "SG_NDVI", "regular_NDVI", "regular_EVI"]

best_on_test_confusion = pd.DataFrame(columns=columns_, index=np.arange(len(labels_)))
best_on_test_confusion.label = labels_
best_on_test_confusion

# %%
regular_EVI

# %%
best_on_test

# %%
for a_type in columns_[1:]:
    curr_min = best_on_test[a_type].min()
    ratio_ = list(best_on_test[best_on_test[a_type]==curr_min].ratio)[0]
    file_name = "02_" + a_type + "_TL_count_TFPR_" + ratio_ + ".csv"
    curr_file = pd.read_csv(in_dir + file_name, index_col=0)

    # find columns with minimum errors in them
    columns_with_min_errors=[]
    for a_col in list(curr_file.columns):
        if curr_min in list(curr_file[a_col]):
            columns_with_min_errors += [a_col]

    # among the columns with min error, find the one
    # with minimum false-double. if not unique get the first one!
    max_true_double = -np.inf
    max_true_double_ratio = "a"
    for a_col in columns_with_min_errors:
        if curr_file.loc["True Double", a_col]>max_true_double:
            max_true_double = curr_file.loc["True Double", a_col]
            max_true_double_ratio = a_col

    best_on_test_confusion[a_type] = list(curr_file.loc[best_on_test_confusion.label, max_true_double_ratio])

# %%
best_on_test_confusion

# %%
best_on_test

# %%

# %%
size = 15
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 12, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size*1.2,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)


titlee="error counts for different oversampling ratios (DL)"
fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

(ax1) = axs; ax1.grid(True);

ax1.plot(best_on_test["ratio"], best_on_test["SG_EVI"],      
         linewidth=4, color="k", label="SG_EVI", linestyle="--") 

ax1.plot(best_on_test["ratio"], best_on_test["SG_NDVI"],
         linewidth=4, color="red", label="SG_NDVI", linestyle="dotted") # 

ax1.plot(best_on_test["ratio"], best_on_test["regular_EVI"], 
         linewidth=4, color="c", label="regular_EVI", marker="o", markersize=10) 

ax1.plot(best_on_test["ratio"], best_on_test["regular_NDVI"],
         linewidth=4, color="dodgerblue", label="regular_NDVI", 
         marker="v", markersize=12) # linestyle="dashdot"

ax1.set_title(titlee)
ax1.set_ylabel(f"error count") # , labelpad=20); # fontsize = label_FontSize,
ax1.set_xlabel(f"sampling ratios") # , labelpad=20); # fontsize = label_FontSize,
ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax1.legend(loc="best");
ax1.set_xticklabels(np.arange(3, 9))
####################################


# plt.yticks(np.arange(0, 1.05, 0.2));
# ax.xaxis.set_major_locator(mdates.YearLocator(1))
# ax1.set_ylim(-0.1, 1.05);

plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"
os.makedirs(plot_dir, exist_ok=True)

file_name = plot_dir + "overSample_ratio_DL_err.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "overSample_ratio_DL_err.png"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
best_on_test_accuracy = best_on_test.copy()
best_on_test_accuracy.loc[:, "SG_EVI":"regular_EVI"] = 632-best_on_test_accuracy.loc[:, "SG_EVI":"regular_EVI"]
best_on_test_accuracy.loc[:, "SG_EVI":"regular_EVI"] = (best_on_test_accuracy.loc[:, "SG_EVI":"regular_EVI"]/632)*100

best_on_test_accuracy['SG_EVI'] = best_on_test_accuracy['SG_EVI'].astype(float).round(2)
best_on_test_accuracy['SG_NDVI'] = best_on_test_accuracy['SG_NDVI'].astype(float).round(2)
best_on_test_accuracy['regular_NDVI'] = best_on_test_accuracy['regular_NDVI'].astype(float).round(2)
best_on_test_accuracy['regular_EVI'] = best_on_test_accuracy['regular_EVI'].astype(float).round(2)
best_on_test_accuracy

# %%
size = 15
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 12, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size*1.2,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)


titlee="accuracy for different oversampling ratios (DL)"
fig, axs = plt.subplots(1, 1, figsize=(10, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

(ax1) = axs; ax1.grid(True);

ax1.plot(best_on_test_accuracy["ratio"], best_on_test_accuracy["SG_EVI"],      
         linewidth=4, color="k", label="SG_EVI", linestyle="--") 

ax1.plot(best_on_test_accuracy["ratio"], best_on_test_accuracy["SG_NDVI"],
         linewidth=4, color="red", label="SG_NDVI", linestyle="dotted") # 

ax1.plot(best_on_test_accuracy["ratio"], best_on_test_accuracy["regular_EVI"], 
         linewidth=4, color="c", label="regular_EVI", marker="o", markersize=10) 

ax1.plot(best_on_test_accuracy["ratio"], best_on_test_accuracy["regular_NDVI"],
         linewidth=4, color="dodgerblue", label="regular_NDVI", 
         marker="v", markersize=12) # linestyle="dashdot"

ax1.set_title(titlee)
ax1.set_ylabel(f"accuracy") # , labelpad=20); # fontsize = label_FontSize,
ax1.set_xlabel(f"sampling ratios") # , labelpad=20); # fontsize = label_FontSize,
ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
ax1.legend(loc="best");
ax1.set_xticklabels(np.arange(3, 9))
####################################
# plt.yticks(np.arange(0, 1.05, 0.2));
# ax.xaxis.set_major_locator(mdates.YearLocator(1))
# ax1.set_ylim(-0.1, 1.05);

plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"
os.makedirs(plot_dir, exist_ok=True)

file_name = plot_dir + "overSample_ratio_DL_Acc.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "overSample_ratio_DL_Acc.png"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%

# %%

# %%

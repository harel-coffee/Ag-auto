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

# %%
import shutup, time, random

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
from random import seed, random

import os, os.path, shutil, sys
import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

# %%

# %%
indeks = "NDVI"
model = "DL"

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
data_dir = dir_base + "VI_TS/05_SG_TS/"
param_dir = dir_base + "parameters/"

# %%
data = pd.read_csv(data_dir + "NDVI_SG_intersect_batchNumber14_JFD_pre2008.csv")
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])

# %%
curr_ID = data.ID.unique()[0]

# %%
crr_fld = data[data.ID == curr_ID].copy()
crr_fld.reset_index(drop=True, inplace=True)

yrs = crr_fld.human_system_start_time.dt.year.unique()

# %%
a_year = yrs[1]

# %%
crr_fld_yr = crr_fld[crr_fld.human_system_start_time.dt.year == a_year]
crr_fld_yr.reset_index(drop=True, inplace=True)
fig, ax = plt.subplots()
fig.set_size_inches(10, 2.5)
ax.grid(False)
ax.plot(
    crr_fld_yr["human_system_start_time"], crr_fld_yr[indeks], c="dodgerblue", linewidth=5
)
ax.axis("off")
left = crr_fld_yr["human_system_start_time"][0]
right = crr_fld_yr["human_system_start_time"].values[-1]
ax.set_xlim([left, right]);
# the following line also works
ax.set_ylim([-0.005, 1]);

image_dir = '/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/WSDA//'
image_name = image_dir + "fly_test.jpg"
plt.savefig(fname = image_name, dpi = 200, bbox_inches = "tight", facecolor = "w")

# %%
crr_fld_yr.human_system_start_time[0]

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
winnerModels = pd.read_csv(param_dir + "winnerModels_overSample.csv")
winnerModel = "01_TL_NDVI_SG_train80_Oct17_oversample5.h5"

# %%

# %%
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16
import tensorflow as tf

# from keras.optimizers import SGD
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# %%
model_dir_base = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/overSample/"
ML_model = load_model(model_dir_base + model + "/" + winnerModel)

# %%
p_filenames_clean = crr_fld_yr.ID.unique()[0] + "_" + str(crr_fld_yr.human_system_start_time.dt.year.unique()[0])
predictions = pd.DataFrame({"filename": [p_filenames_clean]})
predictions["prob_single"] = -1.0
predictions

# %%
img = nc.load_image(image_name)

# img = nc.load_image("/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/images_DL_oversample/" + \
#                     "oversample3/regular_groundTruth_images_EVI/double_308_WSDA_SF_2016.jpg")
print (f"{img.shape = }")
ML_model.predict(img, verbose=False)[0][0]

# %%
import PIL
from PIL import Image

# %%
type(fig)


# %%
# Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

# %%
# Image.frombytes('RGB', (224, 224))

# %%
def from_canvas():
    lst = list(fig.canvas.get_width_height())
    lst.append(3)
    return PIL.Image.fromarray(np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(lst))


# %%
fig_arr = img_to_array(fig)

# %%
ML_model.predict(fig, verbose=False)[0][0]

# %%

# %%

# %%

# %%
ML_model

# %%

# %%

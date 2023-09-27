import shutup  # , random

shutup.please()

# from keras.optimizers import SGD
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd
from datetime import date, datetime
from random import seed, random

import sys, os, os.path, shutil, h5py, time
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow
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

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

####################################################################################
###
###      Parameters
###
####################################################################################

VI_idx = sys.argv[1]
smooth_type = sys.argv[2]
train_ID = sys.argv[3]  # we have different training sets: 1, 2, 3, 4, 5, 6
SR = sys.argv[4]  # sample Ratio 3, 4, 5, 6, 7, 8

print("Passed Args. are: ", VI_idx, ",", smooth_type, ",", train_ID, ",", SR)
####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

ML_data_dir_base = data_base + "/ML_data_Oct17/"
overSamp_data_base = ML_data_dir_base + "overSamples/"
model_dir = data_base + "ML_Models_Oct17/DeskReject/"
train_test_dir = overSamp_data_base + "train_test_DL_" + str(train_ID) + "/"

train_plot_dir = (
    train_test_dir + "/oversample" + str(SR) + "/" + smooth_type + "_" + VI_idx + "_train/"
)
test_plot_dir = train_test_dir + VI_idx + "_" + smooth_type + "_test/"
train_test_split_dir = ML_data_dir_base + "train_test_DL_" + str(train_ID) + "/"

res_dir = data_base + "DeskRejectResults/00_predictions/"
os.makedirs(res_dir, exist_ok=True)


#####################################################################
######
######                           Body
######
#####################################################################
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img


test_fName = "test20_split_2Bconsistent_Oct17_DL_" + str(train_ID) + ".csv"
test20 = pd.read_csv(train_test_split_dir + test_fName)
test20["file_name"] = "single"
double_idx = test20[test20.Vote == 2].index
test20.loc[double_idx, "file_name"] = "double"
test20["file_name"] = test20.file_name + "_" + test20.ID + ".jpg"


train_fName = "train80_split_2Bconsistent_Oct17_DL_" + str(train_ID) + ".csv"
train80 = pd.read_csv(train_test_split_dir + train_fName)
train80["VoteLetter"] = "single"
double_idx = train80[train80.Vote == 2].index
train80.loc[double_idx, "VoteLetter"] = "double"
train80["file_name"] = train80.VoteLetter + "_" + train80.ID + "_copy0.jpg"


print(test20.shape)
print(train80.shape)
print("===================================================")

test20["prob_single"] = -1.0
train80["prob_single"] = -1.0

model_name = "_".join(["01_TL", VI_idx, smooth_type, "train80_SR", SR, "DL", train_ID])
model_name = model_name + ".h5"
ML_model = load_model(model_dir + model_name)

for idx in test20.index:
    img = load_image(test_plot_dir + test20.loc[idx, "file_name"])
    test20.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]

del idx

for idx in train80.index:
    img = load_image(
        train_plot_dir + train80.loc[idx, "VoteLetter"] + "/" + train80.loc[idx, "file_name"]
    )
    train80.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]
##############################################################################

train80 = train80[["ID", "Vote", "prob_single"]]
test20 = test20[["ID", "Vote", "prob_single"]]

train80["train_test"] = "train"
test20["train_test"] = "test"

results = pd.concat([train80, test20])

out_name = VI_idx + "_" + smooth_type + "_DL_" + train_ID + "_SR_" + SR + ".csv"
results.to_csv(res_dir + out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

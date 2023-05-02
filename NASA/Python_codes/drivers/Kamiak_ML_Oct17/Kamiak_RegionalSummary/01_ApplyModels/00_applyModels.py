import shutup

shutup.please()
import numpy as np
import pandas as pd
import sys, os, os.path, shutil
from datetime import date, datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

import pickle, h5py
import sys, os, os.path

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
####################################################################################
###
###      Parameters
###
####################################################################################
size = sys.argv[1]
VI_idx = sys.argv[2]
smooth = sys.argv[3]
county_ = sys.argv[4]
model = sys.argv[5]

print(f"Passed Args. are: {size=:}, {VI_idx=:}, {smooth=:}, {county_=:}, {model=:}!")

"""
   Directories
"""
param_dir = "/data/project/agaid/h.noorazar/NASA/parameters/"

dir_base = "/data/project/agaid/h.noorazar/NASA/Data_Models_4_RegionalStat/"
in_dir = dir_base + "01_wideData/"
SF_data_dir = dir_base + "00_SF_dataPart/"
model_dir = dir_base + "00_ML_models_Oct17/"
out_dir = dir_base + "02_ML_preds/"
os.makedirs(out_dir, exist_ok=True)


#####################################################################
######
######                           Body
######
#####################################################################
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img


meta_names = ["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]
SF_data = pd.DataFrame()
for file in meta_names:
    curr_file = pd.read_csv(SF_data_dir + file)
    SF_data = pd.concat([SF_data, curr_file])

###### Filter SF Data so we can use field IDs to filter widen TS.
#### We only want irrigated area, for sure:
county_ = county_.replace("_", " ")
SF_data = SF_data[SF_data.county == county_].copy()
print(f"{'In current county: ', len(SF_data.ID.unique())}")

SF_data = nc.filter_out_nonIrrigated(SF_data)
print(f"{'Irrigated Fields: ', len(SF_data.ID.unique())}")

if size == "large":
    SF_data = SF_data[SF_data.Acres > 10].copy()
    print(f"{SF_data.Acres.min()=}")
else:
    SF_data = SF_data[SF_data.Acres <= 10].copy()
    print(f"{SF_data.Acres.min()=}")
print(f"{'after size filter: ', len(SF_data.ID.unique())}")

winnerModels = pd.read_csv(param_dir + "winnerModels.csv")
wide_TS = pd.read_csv(in_dir + VI_idx + "_" + smooth + "_wide.csv")

print(f"{'wide_TS Fields: ', len(wide_TS.ID.unique())}")
wide_TS = wide_TS[wide_TS.ID.isin(list(SF_data.ID.unique()))]
print(f"{'wide_TS Fields: ', len(wide_TS.ID.unique())}")

wide_TS.reset_index(drop=True, inplace=True)

winnerModel = np.array(
    winnerModels.loc[
        (winnerModels.VI_idx == VI_idx)
        & (winnerModels.smooth == smooth)
        & (winnerModels.model == model)
    ].output_name
)[0]
print(f"{winnerModel=}")
##
##    Read Model
##
if winnerModel.endswith(".sav"):
    ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    predictions = ML_model.predict(wide_TS.iloc[:, 1:])
    pred_colName = model + "_" + VI_idx + "_" + smooth + "_preds"
    A = pd.DataFrame(columns=["ID", pred_colName])
    A.ID = wide_TS.ID.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A
else:
    # from keras.utils import to_categorical
    from tensorflow.keras.utils import to_categorical, load_img, img_to_array
    from keras.models import Sequential, Model, load_model
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf

    # from keras.optimizers import SGD
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator

    ML_model = load_model(model_dir + winnerModel)

    prob_thresholds = [
        3,
        3.4,
        3.5,
        3.6,
        4,
        5,
        6,
        7,
        8,
        9,
        9.1,
        9.2,
        9.3,
        9.4,
        9.5,
        9.6,
        9.7,
        9.8,
        9.9,
    ]

    needed_cols = [
        "predType_point3",
        "predType_point34",
        "predType_point35",
        "predType_point36",
        "predType_point4",
        "predType_point5",
        "predType_point6",
        "predType_point7",
        "predType_point8",
        "predType_point9",
        "predType_point91",
        "predType_point92",
        "predType_point93",
        "predType_point94",
        "predType_point95",
        "predType_point96",
        "predType_point97",
        "predType_point98",
        "predType_point99",
    ]

    plot_dir = dir_base + "01_cleanPlots_4_DL/" + VI_idx + "_" + smooth + "/"
    p_filenames = os.listdir(plot_dir)
    p_filenames_clean = []
    for a_file in p_filenames:
        if a_file.endswith(".jpg"):
            if a_file.split(".")[0] in SF_data.ID.unique():
                p_filenames_clean += [a_file]

    # print ("len(p_filenames_clean) is [{}].".format(len(p_filenames_clean)))

    predictions = pd.DataFrame({"filename": p_filenames_clean})
    predictions["prob_single"] = -1.0

    for idx in predictions.index:
        img = load_image(plot_dir + predictions.loc[idx, "filename"])
        predictions.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]

    for prob in np.divide(prob_thresholds, 10).round(2):
        colName = "prob_point" + str(prob)[2:]
        # print ("line 39: " + str(prob))
        # print ("line 40: " + colName)
        predictions.loc[predictions.prob_single < prob, colName] = "double"
        predictions.loc[predictions.prob_single >= prob, colName] = "single"


######  Export Output
pred_colName = model + "_" + VI_idx + "_" + smooth + "_preds"
out_name = out_dir + pred_colName + "_" + county_.replace(" ", "_") + "_" + size + ".csv"
predictions.to_csv(out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

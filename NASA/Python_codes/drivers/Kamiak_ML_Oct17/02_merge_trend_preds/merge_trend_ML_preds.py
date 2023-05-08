import shutup, time  # , random

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
import sys, os, os.path, shutil
import pickle, h5py

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")


####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"
in_dir = data_base + "trend_ML_preds/"

out_dir = data_base + "merged_trend_ML_preds/"
os.makedirs(out_dir, exist_ok=True)
####################################################################################
###
###      Read
###
####################################################################################

##
##    Read Model
##
if winnerModel.endswith(".sav"):
    f_name = VI_idx + "_" + smooth + "_intersect_batchNumber" + batch_no + "_wide_JFD.csv"
    wide_TS = pd.read_csv(in_dir + f_name)
    print("wide_TS.shape: ", wide_TS.shape)

    ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    predictions = ML_model.predict(wide_TS.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx + "_" + smooth + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = wide_TS.ID.values
    A.year = wide_TS.year.values
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

    plot_dir = in_dir
    # p_filenames = os.listdir(plot_dir)

    f_name = "NDVI_SG_intersect_batchNumber" + batch_no + "_wide_JFD.csv"
    wide_TS = pd.read_csv(data_base + "VI_TS/05_SG_TS/" + f_name)
    p_filenames_clean = list(wide_TS.ID + "_" + wide_TS.year.astype(str) + ".jpg")

    # p_filenames_clean = []
    # for a_file in p_filenames:
    #     if a_file.endswith(".jpg"):
    #         # if a_file.split(".")[0] in SF_data.ID.unique():
    #         p_filenames_clean += [a_file]

    # print ("len(p_filenames_clean) is [{}].".format(len(p_filenames_clean)))

    predictions = pd.DataFrame({"filename": p_filenames_clean})
    predictions["prob_single"] = -1.0

    for idx in predictions.index:
        img = load_image(plot_dir + predictions.loc[idx, "filename"])
        predictions.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]

    # for prob in np.divide(prob_thresholds, 10).round(2):
    #     colName = "prob_point" + str(prob)[2:]
    #     # print ("line 39: " + str(prob))
    #     # print ("line 40: " + colName)
    #     predictions.loc[predictions.prob_single < prob, colName] = "d"
    #     predictions.loc[predictions.prob_single >= prob, colName] = "s"


######  Export Output
pred_colName = VI_idx + "_" + smooth + "_" + model + "_batchNumber" + batch_no + "_preds"
out_name = out_dir + pred_colName + ".csv"
predictions.to_csv(out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

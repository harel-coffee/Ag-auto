import numpy as np
import pandas as pd
from datetime import date, datetime
import time

import sys, os, os.path
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc
import NASA_plot_core as rcp

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
####################################################################################
###
###      Parameters
###
####################################################################################
VI_idx = sys.argv[1]
smooth_type = sys.argv[2]
print(f"Passed Args. are: {VI_idx=:}, {smooth_type=:}!")

#####    Directories
dir_base = "/data/project/agaid/h.noorazar/NASA/Data_Models_4_RegionalStat/"
in_dir = dir_base + "00_SmoothedData/"
SF_data_dir = dir_base + "00_SF_dataPart/"
out_dir = dir_base + "01_cleanPlots_4_DL/" + VI_idx + "_" + smooth_type + "/"
os.makedirs(out_dir, exist_ok=True)


######### Read SF data

meta_names = ["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]
SF_data = pd.DataFrame()
for file in meta_names:
    curr_file = pd.read_csv(SF_data_dir + file)
    SF_data = pd.concat([SF_data, curr_file])

SF_data = nc.filter_out_nonIrrigated(SF_data)
print(f"{'Irrigated Fields: ', len(SF_data.ID.unique())}")

######### Read Time Series

file_names = [
    smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv",
    smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv",
    smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv",
    smooth_type + "_FranklinYakima2018_" + VI_idx + "_JFD.csv",
]

data = pd.DataFrame()
for file in file_names:
    curr_file = pd.read_csv(in_dir + file)
    curr_file["human_system_start_time"] = pd.to_datetime(curr_file["human_system_start_time"])

    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years) == 3 or len(all_years) == 2:
        proper_year = all_years[1]
    elif len(all_years) == 1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year == proper_year]
    data = pd.concat([data, curr_file])

#### Subset to irrigated fields
print(f"{SF_data.Irrigtn.unique()=}")
print("data shape before irrigation filtering is", SF_data.shape)
data = data[data.ID.isin(list(SF_data.ID.unique()))]
print("data shape after irrigation filtering is", data.shape)


data.reset_index(drop=True, inplace=True)
data.loc[data[VI_idx] < 0, VI_idx] = 0
data.head(2)

###
###  Plot
###
for curr_ID in data.ID.unique():
    crr_fld = data[data.ID == curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)

    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 2.5)
    ax.grid(False)
    ax.plot(crr_fld["human_system_start_time"], crr_fld[VI_idx], c="dodgerblue", linewidth=5)
    ax.axis("off")
    left = crr_fld["human_system_start_time"][0]
    right = crr_fld["human_system_start_time"].values[-1]
    ax.set_xlim([left, right])

    # the following line alsow orks
    ax.set_ylim([-0.005, 1])

    # train_images is the same as expert labels!
    fig_name = out_dir + curr_ID + ".jpg"
    plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight", facecolor="w")
    plt.close("all")

print(plot_path)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

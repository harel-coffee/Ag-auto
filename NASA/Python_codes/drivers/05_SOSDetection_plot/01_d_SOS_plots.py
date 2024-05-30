####
#### Nov 8, 2021
####


# import matplotlib.backends.backend_pdf
import csv
import numpy as np
import pandas as pd

# from IPython.display import Image
import datetime
from datetime import date
import time
import scipy
import scipy.signal
import os, os.path
import matplotlib

from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import sys

start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append("/home/hnoorazar/NASA/")
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters
###
####################################################################################
eleven_colors = [
    "gray",
    "lightcoral",
    "red",
    "peru",
    "darkorange",
    "gold",
    "olive",
    "green",
    "blue",
    "violet",
    "deepskyblue",
]

# indeks = sys.argv[1]
randCount = sys.argv[1]
if randCount != "all":
    randCount = int(randCount)

print("randCount is:")
print(randCount)
print(type(randCount))

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
raw_dir = "/data/hydro/users/Hossein/NASA/01_raw_GEE/"
data_dir = "/data/hydro/users/Hossein/NASA/05_SG_TS/"
SOS_plot_dir = "/data/hydro/users/Hossein/NASA/06_SOS_plots/"

print("_________________________________________________________")
print("data dir is: " + data_dir)
print("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################
if randCount == "all":
    f_name_NDVI = "04_SG_int_Grant_Irr_2008_2018_NDVI.csv"
    f_name_EVI = "04_SG_int_Grant_Irr_2008_2018_EVI.csv"
    SG_df_NDVI = pd.read_csv(data_dir + f_name_NDVI, low_memory=False)
    SG_df_EVI = pd.read_csv(data_dir + f_name_EVI, low_memory=False)

    # convert the strings to datetime format
    SG_df_NDVI["human_system_start_time"] = pd.to_datetime(
        SG_df_NDVI["human_system_start_time"]
    )
    SG_df_EVI["human_system_start_time"] = pd.to_datetime(
        SG_df_EVI["human_system_start_time"]
    )

    plot_path = SOS_plot_dir + "allfields/"
    os.makedirs(plot_path, exist_ok=True)
    print("plot_path is: " + plot_path)
    print("_________________________________________________________")
else:
    f_name_NDVI = (
        "04_SG_int_Grant_Irr_2008_2018_"
        + "NDVI"
        + "_"
        + str(randCount)
        + "randomfields.csv"
    )
    f_name_EVI = (
        "04_SG_int_Grant_Irr_2008_2018_"
        + "EVI"
        + "_"
        + str(randCount)
        + "randomfields.csv"
    )
    SG_df_NDVI = pd.read_csv(data_dir + f_name_NDVI, low_memory=False)
    SG_df_EVI = pd.read_csv(data_dir + f_name_EVI, low_memory=False)

    # convert the strings to datetime format
    SG_df_NDVI["human_system_start_time"] = pd.to_datetime(
        SG_df_NDVI["human_system_start_time"]
    )
    SG_df_EVI["human_system_start_time"] = pd.to_datetime(
        SG_df_EVI["human_system_start_time"]
    )

    plot_path = SOS_plot_dir + str(randCount) + "randomfields/"
    os.makedirs(plot_path, exist_ok=True)
    print("plot_path is: " + plot_path)
    print("_________________________________________________________")

f_names = [
    "L5_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_102008-01-01_2012-05-05.csv",
    "L7_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10_2008-01-01_2021-09-23.csv",
    "L8_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10_2008-01-01_2021-10-14.csv",
]

L5 = pd.read_csv(raw_dir + f_names[0], low_memory=False)
L7 = pd.read_csv(raw_dir + f_names[1], low_memory=False)
L8 = pd.read_csv(raw_dir + f_names[2], low_memory=False)

# L5_EVI = L5.copy()
# L7_EVI = L7.copy()
# L8_EVI = L8.copy()

# L5_NDVI = L5.copy()
# L7_NDVI = L7.copy()
# L8_NDVI = L8.copy()

IDs = np.sort(SG_df_NDVI["ID"].unique())
### List of unique fields
print("_____________________________________")
print("len(IDs)")
print(len(IDs))
print("_____________________________________")


raw_df = pd.concat([L5, L7, L8])
del (L5, L7, L8)

raw_df = raw_df[raw_df.ID.isin(IDs)]

raw_df_EVI = raw_df.copy()
raw_df_NDVI = raw_df.copy()
del raw_df

raw_df_EVI.drop(["NDVI"], axis=1, inplace=True)
raw_df_NDVI.drop(["EVI"], axis=1, inplace=True)

raw_df_EVI = raw_df_EVI[raw_df_EVI["EVI"].notna()]
raw_df_NDVI = raw_df_NDVI[raw_df_NDVI["NDVI"].notna()]

raw_df_EVI = nc.add_human_start_time_by_system_start_time(raw_df_EVI)
raw_df_NDVI = nc.add_human_start_time_by_system_start_time(raw_df_NDVI)

########################################

SG_df_NDVI = nc.initial_clean(df=SG_df_NDVI, column_to_be_cleaned="NDVI")
SG_df_EVI = nc.initial_clean(df=SG_df_EVI, column_to_be_cleaned="EVI")

raw_df_NDVI = nc.initial_clean(df=raw_df_NDVI, column_to_be_cleaned="NDVI")
raw_df_EVI = nc.initial_clean(df=raw_df_EVI, column_to_be_cleaned="EVI")

counter = 0

for ID in IDs:
    if counter % 100 == 0:
        print("_____________________________________")
        print("counter: " + str(counter))
        print(ID)

    curr_SG_NDVI = SG_df_NDVI[SG_df_NDVI["ID"] == ID].copy()
    curr_SG_NDVI.sort_values(by=["human_system_start_time"], inplace=True)
    curr_SG_NDVI.reset_index(drop=True, inplace=True)

    curr_raw_NDVI = raw_df_NDVI[raw_df_NDVI["ID"] == ID].copy()
    curr_raw_NDVI.sort_values(by=["human_system_start_time"], inplace=True)
    curr_raw_NDVI.reset_index(drop=True, inplace=True)

    curr_SG_EVI = SG_df_EVI[SG_df_EVI["ID"] == ID].copy()
    curr_SG_EVI.sort_values(by=["human_system_start_time"], inplace=True)
    curr_SG_EVI.reset_index(drop=True, inplace=True)

    curr_raw_EVI = raw_df_EVI[raw_df_EVI["ID"] == ID].copy()
    curr_raw_EVI.sort_values(by=["human_system_start_time"], inplace=True)
    curr_raw_EVI.reset_index(drop=True, inplace=True)
    ################################################################

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(18, 6),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    (ax1, ax2) = axs
    ax1.grid(True)
    ax2.grid(True)
    # ax3.grid(True); ax4.grid(True); ax5.grid(True); ax6.grid(True);

    # Plot NDVIs
    ncp.SG_clean_SOS_orchardinPlot(
        raw_dt=curr_raw_NDVI,
        SG_dt=curr_SG_NDVI,
        idx="NDVI",
        ax=ax1,
        onset_cut=0.3,
        offset_cut=0.3,
    )

    # ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_NDVI,
    #                                SG_dt = curr_SG_NDVI,
    #                                idx = "NDVI",
    #                                ax = ax2,
    #                                onset_cut = 0.4,
    #                                offset_cut = 0.4);

    # ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_NDVI,
    #                                SG_dt = curr_SG_NDVI,
    #                                idx = "NDVI",
    #                                ax = ax3,
    #                                onset_cut = 0.5,
    #                                offset_cut = 0.5);

    # Plot EVIs

    ncp.SG_clean_SOS_orchardinPlot(
        raw_dt=curr_raw_EVI,
        SG_dt=curr_SG_EVI,
        idx="EVI",
        ax=ax2,
        onset_cut=0.3,
        offset_cut=0.3,
    )

    # ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_EVI,
    #                                SG_dt = curr_SG_EVI,
    #                                idx = "EVI",
    #                                ax = ax5,
    #                                onset_cut = 0.4,
    #                                offset_cut = 0.4);

    # ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_EVI,
    #                                SG_dt = curr_SG_EVI,
    #                                idx = "EVI",
    #                                ax = ax6,
    #                                onset_cut = 0.5,
    #                                offset_cut = 0.5);

    fig_name = plot_path + ID + ".pdf"

    os.makedirs(plot_path, exist_ok=True)

    plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")
    plt.close("all")
    counter += 1


print("done")
end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))

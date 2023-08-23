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

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################
start_time = time.time()

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

try:
    print("numpy.__version__=", numpy.__version__)
except:
    print("numpy.__version__ not printed")

####################################################################################
###
###      Parameters
###
####################################################################################

indeks = sys.argv[1]
smooth_type = sys.argv[2]
batch_no = str(sys.argv[3])
print(f"Passed Args. are: {indeks=:}, {smooth_type=:}, and {batch_no=:}!")

####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

if smooth_type == "regular":
    in_dir = data_base + "VI_TS/04_regularized_TS/"
else:
    in_dir = data_base + "VI_TS/05_SG_TS/"

out_dir = data_base + "06_cleanPlots_4_DL_pre2008/" + indeks + "_" + smooth_type + "_plots/"
os.makedirs(out_dir, exist_ok=True)
####################################################################################
print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = indeks + "_" + smooth_type + "_intersect_batchNumber" + str(batch_no) + "_JFD_pre2008.csv"

data = pd.read_csv(in_dir + f_name)
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])

counter = 0
for curr_ID in data.ID.unique():
    if counter == 0:
        print(f"{curr_ID=:}")
    crr_fld = data[data.ID == curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)

    yrs = crr_fld.human_system_start_time.dt.year.unique()

    for a_year in yrs:
        if counter == 0:
            print(f"{a_year=:}")
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
        ax.set_xlim([left, right])
        # the following line also works
        ax.set_ylim([-0.005, 1])
        fig_name = out_dir + curr_ID + "_" + str(a_year) + ".jpg"
        plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight", facecolor="w")
        if counter == 0:
            print("Line 88")
            counter += 1
        plt.close("all")

print(plot_path)

end_time = time.time()
print("current time is {}".format(time.time()))
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))


print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

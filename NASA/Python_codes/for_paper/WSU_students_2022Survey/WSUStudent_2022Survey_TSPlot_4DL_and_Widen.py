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
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
VI_TS_dir = dir_base + "VI_TS/09_WSU_students2022Survey/"
plot_dir_base = VI_TS_dir + "plots/"
os.makedirs(plot_dir_base, exist_ok=True)

# %%
indekses = ["EVI", "NDVI"]
smooths = ["regular", "SG"]

# %%
for VI_indeks in indekses:
    for smooth_ in smooths:
        print (f"{VI_indeks=}, {smooth_=}")
        TS_f_name = VI_indeks + "_" + smooth_ + "_WSUStudentSurvey2022.csv"
        TS_df = pd.read_csv(VI_TS_dir + TS_f_name)
        TS_df["human_system_start_time"] = pd.to_datetime(TS_df['human_system_start_time'])
        counter = 0
        for curr_ID in TS_df.ID.unique():
            if counter == 0:
                print(f"{curr_ID=:}")
                counter += 1
                print ()
            crr_fld = TS_df[TS_df.ID == curr_ID].copy()
            crr_fld.reset_index(drop=True, inplace=True)

            yrs = crr_fld.human_system_start_time.dt.year.unique()

            for a_year in yrs:
                crr_fld_yr = crr_fld[crr_fld.human_system_start_time.dt.year == a_year]
                crr_fld_yr.reset_index(drop=True, inplace=True)
                fig, ax = plt.subplots()

                fig.set_size_inches(10, 2.5)
                ax.grid(False)
                ax.plot(
                    crr_fld_yr["human_system_start_time"], crr_fld_yr[VI_indeks], c="dodgerblue", linewidth=5
                )
                ax.axis("off")

                left = crr_fld_yr["human_system_start_time"][0]
                right = crr_fld_yr["human_system_start_time"].values[-1]
                ax.set_xlim([left, right])

                # the following line also works
                ax.set_ylim([-0.005, 1])

                out_dir = plot_dir_base + VI_indeks + "_" + smooth_ + "/"
                os.makedirs(out_dir, exist_ok=True)
                fig_name = out_dir + curr_ID + "_" + str(a_year) + ".jpg"
                plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight", facecolor="w")
                plt.close("all")

print ("done")

# %% [markdown]
# # Widen

# %%
for VI_indeks in indekses:
    for smooth_ in smooths:
        print (f"{VI_indeks=}, {smooth_=}")
        print ()
        TS_f_name = VI_indeks + "_" + smooth_ + "_WSUStudentSurvey2022.csv"
        TS_df = pd.read_csv(VI_TS_dir + TS_f_name)
        TS_df["human_system_start_time"] = pd.to_datetime(TS_df['human_system_start_time'])
        counter = 0
        
        VI_colnames = [VI_indeks + "_" + str(ii) for ii in range(1, 37)]
        columnNames = ["ID", "year"] + VI_colnames

        years = TS_df.human_system_start_time.dt.year.unique()
        IDs = TS_df.ID.unique()
        no_rows = len(IDs) * len(years)

        data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
        data_wide.ID = list(IDs) * len(years)
        data_wide.sort_values(by=["ID"], inplace=True)
        data_wide.reset_index(drop=True, inplace=True)
        data_wide.year = list(years) * len(IDs)


        for an_ID in IDs:
            curr_field = TS_df[TS_df.ID == an_ID]
            curr_years = curr_field.human_system_start_time.dt.year.unique()
            for a_year in curr_years:
                curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]

                data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

                if VI_indeks == "EVI":
                    data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
                elif VI_indeks == "NDVI":
                    data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

        data_wide.drop_duplicates(inplace=True)
        data_wide.dropna(inplace=True)
        
        out_name = VI_TS_dir + VI_indeks + "_" + smooth_ + "_WSUStudentSurvey2022_wide_JFD.csv"
        data_wide.sort_values(by=["ID"], inplace=True)
        data_wide.to_csv(out_name, index=False)

print ("done")

# %%
data_wide.head(2)

# %%
out_name

# %%

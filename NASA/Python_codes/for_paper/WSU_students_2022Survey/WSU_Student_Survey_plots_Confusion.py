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
import pandas as pd
import numpy as np
import datetime, time, re
from datetime import date

import os, os.path, sys
from os import listdir
from os.path import isfile, join

# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/2022_survey/WSU_Students_2022_GEE/"
TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/09_WSU_students2022Survey/"
pred_dir = TS_dir + "predictions/"

# %%
VIs = ["NDVI", "EVI"]
smooths = ["regular", "SG"]

# %%
meta = pd.read_csv(meta_dir + "WSU_Students_2022_GEE_data.csv")

# %%
double_or_not = pd.read_csv(TS_dir+"WSUStudent2022SurveyCrops_GT_large_importantData.csv")
IDs = double_or_not.ID.unique()

# %%
size = 15
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 15, # medium, large
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

# %%
L7 = pd.read_csv(TS_dir + "L7_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv")
L8 = pd.read_csv(TS_dir+ "L8_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv")
L8.head(2)

# %%
# raw = NDVI_SG_summary_L = pd.merge(L7, L8, on=(["GlobalID"]), how='left')
raw = NDVI_SG_summary_L = pd.concat([L7, L8])
raw.rename(columns={"GlobalID": "ID"}, inplace=True)

raw_NDVI = raw[["ID", "NDVI", "system_start_time"]].copy()
raw_EVI = raw[["ID", "EVI", "system_start_time"]].copy()

raw_NDVI.dropna(subset=["NDVI"], inplace=True)
raw_EVI.dropna(subset=["EVI"], inplace=True)

raw_NDVI.head(2)

# %%
raw_NDVI = nc.add_human_start_time_by_system_start_time(raw_NDVI)
raw_EVI = nc.add_human_start_time_by_system_start_time(raw_EVI)

# %%
raw_NDVI.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
raw_EVI.sort_values(by=["ID", 'human_system_start_time'], inplace=True)


raw_EVI.head(2)

# %%
raw_NDVI.reset_index(drop=True, inplace=True)
raw_EVI.reset_index(drop=True, inplace=True)

# %%
raw_EVI.head(2)

# %%
# raw_NDVI[raw_NDVI.ID=="2dd77d1a-1bd8-401d-abff-b52e3f5000a0"]
ID = "2dd77d1a-1bd8-401d-abff-b52e3f5000a0"

# %%
double_or_not[double_or_not.ID=="2dd77d1a-1bd8-401d-abff-b52e3f5000a0"]

# %%
VI="NDVI"
smooth="regular"
ID=="2dd77d1a-1bd8-401d-abff-b52e3f5000a0"


f_name = VI + "_" + smooth + "_WSUStudentSurvey2022.csv"
TS_df = pd.read_csv(TS_dir + f_name)

TS_df["human_system_start_time"] = pd.to_datetime(TS_df["human_system_start_time"])
TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
print (ID)
if VI == "EVI":
    curr_raw = raw_EVI[raw_EVI.ID==ID]
else:
    curr_raw = raw_NDVI[raw_NDVI.ID==ID]

# %%
curr_df = TS_df[TS_df.ID==ID]
print (ID)
fig, ax = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

ax.grid(which='major', axis='y', linestyle='--')
ax.plot(curr_df["human_system_start_time"], curr_df[VI], 
        linewidth=4, color="dodgerblue", label="smoothed");

ax.scatter(curr_raw["human_system_start_time"],
           curr_raw[VI],s=7,
           c="red",label="raw")

title_source = double_or_not[double_or_not.ID==ID]
plant_count = title_source.plant_count.values[0]
titlee = [title_source.Irrigation.values[0],
          title_source.FirstSurve.values[0],
          title_source.FirstSur_1.values[0],
          title_source.SecondSurv.values[0],
          title_source.SecondSu_1.values[0],
          f'{plant_count=}' ]
if type(title_source.hard.values[0])==str:
    titlee += [title_source.hard.values[0]]
titlee = ", ".join([str(item) for item in titlee])

ax.set_title(titlee)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylim([-0.1, 1.15]);

plot_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/09_WSU_students2022Survey/plots/"
plot_dir = plot_dir_base + VI + "_" + smooth + "/" + plant_count
if type(title_source.hard.values[0])==str:
    plot_dir = plot_dir + "_" + title_source.hard.values[0] + "/"
else:
    plot_dir = plot_dir + "/"

os.makedirs(plot_dir, exist_ok=True)
file_name = plot_dir + ID + ".pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);
plt.show()
plt.close('all')

# %%

# %%
double_or_not[double_or_not.ID==ID]

# %%
VI = VIs[0]
smooth = smooths[0]

for VI in VIs:
    for smooth in smooths:
        print (f"{VI=}, {smooth=}")
        f_name = VI + "_" + smooth + "_WSUStudentSurvey2022.csv"
        TS_df = pd.read_csv(TS_dir + f_name)

        TS_df["human_system_start_time"] = pd.to_datetime(TS_df["human_system_start_time"])
        TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
        for ID in IDs:
            if VI == "EVI":
                curr_raw = raw_EVI[raw_EVI.ID==ID]
            else:
                curr_raw = raw_NDVI[raw_NDVI.ID==ID]
            
            curr_df = TS_df[TS_df.ID==ID]
            fig, ax = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                                   gridspec_kw={'hspace': 0.35, 'wspace': .05});

            ax.grid(which='major', axis='y', linestyle='--')
            ax.plot(curr_df["human_system_start_time"], curr_df[VI], 
                    linewidth=4, color="dodgerblue", label="smoothed");

            ax.scatter(curr_raw["human_system_start_time"],
                       curr_raw[VI],s=7,
                       c="red",label="raw")

            title_source = double_or_not[double_or_not.ID==ID]
            plant_count = title_source.plant_count.values[0]
            titlee = [title_source.Irrigation.values[0],
                      title_source.FirstSurve.values[0],
                      title_source.FirstSur_1.values[0],
                      title_source.SecondSurv.values[0],
                      title_source.SecondSu_1.values[0],
                      f'{plant_count=}' ]
            if type(title_source.hard.values[0])==str:
                titlee += [title_source.hard.values[0]]
            titlee = ", ".join([str(item) for item in titlee])

            ax.set_title(titlee)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.set_ylim([-0.1, 1.15]);

            plot_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/09_WSU_students2022Survey/plots/"
            plot_dir = plot_dir_base + VI + "_" + smooth + "/" + plant_count
            if type(title_source.hard.values[0])==str:
                plot_dir = plot_dir + "_" + title_source.hard.values[0] + "/"
            else:
                plot_dir = plot_dir + "/"

            os.makedirs(plot_dir, exist_ok=True)

            file_name = plot_dir + ID + ".pdf"
            plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);
            plt.close('all')
            # plt.show()

# %%
os.listdir(pred_dir)

# %%
for VI is ["NDVI", "EVI"]:
    for smooth in ["SG", "regular"]:
        for ML in ["DL", "KNN", "SVM", "RF"]:
            curr_preds = pd.read_csv(pred_dir+"NDVI_SG_KNN_preds.csv")
            if ML=="DL":
                newName = ML + "_" + VI + "_" + smooth + "_pSingle"
                curr_preds.rename(columns={"prob_single": newName}, inplace=True)

# %%

# %%

# %%
# double_or_not

# %%

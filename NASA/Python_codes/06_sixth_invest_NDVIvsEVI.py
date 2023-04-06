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
import csv
import numpy as np
import pandas as pd
from math import factorial

import datetime
from datetime import date
import datetime
import time

import scipy
import os, os.path
from os import listdir
from os.path import isfile, join

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sb

import sys


# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %% [markdown]
# ### Set up directories

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/sixth_investig_intersected/"

# %%
L5 = pd.read_csv(data_dir + "L5_T1C2L2_Scaled_intGrant_2008-01-01_2012-05-05.csv")
L7 = pd.read_csv(data_dir + "L7_T1C2L2_Scaled_intGrant_2008-01-01_2021-09-23.csv")
L8 = pd.read_csv(data_dir + "L8_T1C2L2_Scaled_intGrant_2008-01-01_2021-10-14.csv")

# %%
IDs = np.sort(L5.ID.unique())

# %%
L578 = pd.concat([L5, L7, L8])
# del(L5, L7, L8)

# %%
import random

random.seed(10)
np.random.seed(10)

# %% [markdown]
# ## Pick 100 random fields.

# %%
rand_lst = list(np.random.choice(IDs, 100))

# %%
L578 = L578[L578.ID.isin(rand_lst)].copy()
L578.shape

# %%
L578.head(2)

# %%
size = 20
title_FontSize = 10
legend_FontSize = 14
tick_FontSize = 18
label_FontSize = 14

params = {'legend.fontsize': 17,
          'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
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
# pylab.rcParams.update(params)
# plt.rc('text', usetex=True)


# %%
for an_ID in rand_lst:
    curr_NDVI = L578[L578.ID == an_ID].copy()
    curr_NDVI.drop(['EVI'], axis=1, inplace=True)
    curr_NDVI = curr_NDVI[curr_NDVI['NDVI'].notna()]
    curr_NDVI = nc.add_human_start_time_by_system_start_time(curr_NDVI)
    
    curr_EVI = L578[L578.ID == an_ID].copy()
    curr_EVI.drop(['NDVI'], axis=1, inplace=True)
    curr_EVI = curr_EVI[curr_EVI['EVI'].notna()]
    curr_EVI = nc.add_human_start_time_by_system_start_time(curr_EVI)
    
    curr_NDVI.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    curr_NDVI.reset_index(drop=True, inplace=True)
    
    curr_EVI.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    curr_EVI.reset_index(drop=True, inplace=True)
    
    if curr_NDVI.shape[0] == 0: 
        print (an_ID)
    
    if curr_NDVI.shape[0] > 0:
        if curr_EVI.shape[0] > 0:
            ##################################
            ################################## Remove boundary violations
            ################################## This does not happen in NDVI! Just sanity check
            ##################################
            curr_NDVI.loc[curr_NDVI['NDVI'] > 1, "NDVI"]  = 1
            curr_NDVI.loc[curr_NDVI['NDVI'] < -1, "NDVI"] = -1

            #####
            #####   Clip or Remove or Interpolate
            #####
            # curr_EVI.loc[curr_EVI['EVI'] > 1, "EVI"]  = 1
            # curr_EVI.loc[curr_EVI['EVI'] < -1, "EVI"] = -1
            
            curr_EVI.drop(curr_EVI[curr_EVI.EVI > 1].index, inplace=True)
            curr_EVI.drop(curr_EVI[curr_EVI.EVI < -1].index, inplace=True)
            
            curr_NDVI.reset_index(drop=True, inplace=True)
            curr_EVI.reset_index(drop=True, inplace=True)
            
            ##################################
            ################################## Correct big jumps
            ##################################
            NDVI_NoJump = nc.correct_big_jumps_1DaySeries(dataTMS_jumpie = curr_NDVI, 
                                                          give_col = "NDVI", 
                                                          maxjump_perDay = 0.018)

            EVI_NoJump = nc.correct_big_jumps_1DaySeries(dataTMS_jumpie = curr_EVI, 
                                                         give_col = "EVI", 
                                                         maxjump_perDay = 0.018)

            NDVI_NoJump.reset_index(drop=True, inplace=True)
            EVI_NoJump.reset_index(drop=True, inplace=True)

            ##################################
            ################################## Set Negatives to zero
            ##################################
            NDVI_NoJump.loc[NDVI_NoJump['NDVI'] < 0, "NDVI"] = 0
            EVI_NoJump.loc[EVI_NoJump['EVI'] < 0, "EVI"] = 0

            ##################################
            ################################## Regularize (10-day composite) and do SG
            ##################################
            step_size = 10

            NDVI_NoJump['dataset'] = 'L578'
            NDVI_NoJump = nc.regularize_a_field(a_df = NDVI_NoJump, 
                                               V_idks = 'NDVI', 
                                               interval_size = step_size)

            NDVI_NoJump = nc.fill_theGap_linearLine(NDVI_NoJump, V_idx='NDVI')


            EVI_NoJump['dataset'] = 'L578'
            EVI_NoJump = nc.regularize_a_field(a_df = EVI_NoJump, 
                                               V_idks = 'EVI', 
                                               interval_size = step_size)

            EVI_NoJump = nc.fill_theGap_linearLine(EVI_NoJump, V_idx='EVI')

            SG = scipy.signal.savgol_filter(NDVI_NoJump['NDVI'].values, window_length=7, polyorder=3)
            SG[SG > 1 ] = 1     # SG might violate the boundaries. clip them:
            SG[SG < -1 ] = -1
            NDVI_NoJump['NDVI'] = SG    

            SG = scipy.signal.savgol_filter(EVI_NoJump['EVI'].values, window_length=7, polyorder=3)
            SG[SG > 1 ] = 1     # SG might violate the boundaries. clip them:
            SG[SG < -1 ] = -1
            EVI_NoJump['EVI'] = SG

            ##########
            ##########
            ##########
            fig, axs = plt.subplots(2, 1, figsize=(40, 8),
                                sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0.2, 'wspace': .1});

            (ax1, ax2) = axs;
            ax1.grid(True); ax2.grid(True)

            ax1.plot(curr_NDVI['human_system_start_time'], curr_NDVI['NDVI'], '-', label = "raw NDVI", 
                    linewidth=2, color='dodgerblue')

            ax1.plot(curr_EVI['human_system_start_time'], curr_EVI['EVI'], '-', label = "raw EVI", 
                    linewidth=2, color='red')

            ax2.plot(NDVI_NoJump['human_system_start_time'], NDVI_NoJump['NDVI'], '-', label = "SG NDVI", 
                    linewidth=2, color='dodgerblue')

            ax2.plot(EVI_NoJump['human_system_start_time'], EVI_NoJump['EVI'], '-', label = "SG EVI", 
                    linewidth=2, color='red')

            ax1.xaxis.set_major_locator(mdates.YearLocator(1)) # every year.
            ax2.xaxis.set_major_locator(mdates.YearLocator(1)) # every year.
            ax1.set_ylim(-1, 1)
            ax2.set_ylim(-1, 1)
            ax1.legend(loc="upper left");
            ax2.legend(loc="upper left");

            plot_dir = data_dir + "NDVI_vs_EVI/"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            file_name = plot_dir + str(an_ID) + ".pdf"
            # plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)
            plt.close()

# %%

# %%

# %%

# %%

# %%

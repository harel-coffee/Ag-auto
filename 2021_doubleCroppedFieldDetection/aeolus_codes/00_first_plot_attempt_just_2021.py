import matplotlib.backends.backend_pdf
import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import datetime
from datetime import date
import time
import scipy
import scipy.signal
import os, os.path
import matplotlib

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
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
###                      Local
###
####################################################################################
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################
sys.path.append('/home/hnoorazar/NASA/')

import NASA_core as nc
import NASA_plot_core as npc
####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_dir = "/data/hydro/users/Hossein/2021_doubleCroppedFieldDetection/01_idx/"
plot_dir_base = data_dir + "figures_just_2021/"
print ("plot_dir_base is " + plot_dir_base)

os.makedirs(plot_dir_base, exist_ok=True)
os.makedirs(plot_dir_base, exist_ok=True)

print ("_________________________________________________________")
print ("data dir is:")
print (data_dir)
print ("_________________________________________________________")
print ("output_dir is:")
print (plot_dir_base)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################
dataframe_list = []

file_names = ['AdamsFrankBentonGrant_IrrAnn_SH2016_IY_2021.csv',
              'AdamsFrankBentonGrant_IrrAnn_SH2017_IY_2021.csv',
              'AdamsFrankBentonGrant_IrrAnn_SH2018_IY_2021.csv']
A = pd.read_csv(data_dir + file_names[0])
A = A[A['NDVI'].notna()]
dataframe_list.append(A)

A = pd.read_csv(data_dir + file_names[1])
A = A[A['NDVI'].notna()]
dataframe_list.append(A)

A = pd.read_csv(data_dir + file_names[2])
A = A[A['NDVI'].notna()]
dataframe_list.append(A)

all_data = pd.concat(dataframe_list)
all_data.reset_index(drop=True, inplace=True)
all_data = nc.add_human_start_time_by_system_start_time(all_data)
all_data = all_data[all_data.human_system_start_time >= pd.Timestamp('2021-01-01')].copy()
all_data.reset_index(drop=True, inplace=True)
all_data["dataset"] = "Sentinel"

ID_list = list(np.sort(all_data.ID.unique()))

print ("len(ID_list): " + str(len(ID_list)))
##################################################################
##################################################################
####
####  Set the plotting style
####
##################################################################
##################################################################

size = 20

params = {'legend.fontsize': 2,
          'figure.figsize': (3, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.6,
          'ytick.labelsize': size * 0.6,
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


VI="NDVI"
interval_size = 10

color_dict = {'raw': 'r','Sentinel': 'dodgerblue'}
size = 10
tickWidth = 0.6
tickLength = 3
params = {'legend.fontsize': size * 0.5,
          'figure.figsize': (10, 2),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.7,
          'ytick.labelsize': size * 0.7,
          'axes.titlepad': 2,
          'axes.linewidth' : 0.5,
          'xtick.major.size' : tickLength,
          'xtick.major.width': tickWidth, 
          'xtick.minor.size' : tickLength,
          'xtick.minor.width' : tickWidth,
          'ytick.major.size' : tickLength,
          'ytick.major.width': tickWidth, 
          'ytick.minor.size' : tickLength,
          'ytick.minor.width' : tickWidth,
          'legend.loc': 'lower left'}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True

plt.rcParams.update(params)
Lwidth = 1

counter = 0
for curr_field_ID in ID_list:
    if (counter%1000 == 0):
        print ("_____________________________________")
        print ("counter: " + str(counter))
        print (curr_field_ID)

    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.3, 'wspace': .05});
    ax.grid(True)

    curr_dt = all_data[all_data.ID == curr_field_ID].copy()
    curr_dt.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
    npc.one_satellite_smoothed(raw_dt=curr_dt, ax=ax, color_dict=color_dict, idx=VI, time_step_size=interval_size)

    ax.plot(curr_dt['human_system_start_time'], curr_dt['NDVI'], 
            label = "raw", linewidth=Lwidth, color='r')

    assert (len(curr_dt.cntrd_ln.unique()) == 1)
    assert (len(curr_dt.cntrd_lt.unique()) == 1)
    centriod = ", " + str(curr_dt.cntrd_ln.unique()[0]) + ", " + str(curr_dt.cntrd_lt.unique()[0])
    # centriod =  ", " + str(curr_dt.centroid_long.unique()[0]) + "_" + str(curr_dt.centroid_lat.unique()[0] )
    ax.set_title(curr_dt.ID.unique()[0] + ", " + curr_dt.CropTyp.unique()[0] + centriod) 
    ax.set_ylabel('NDVI') # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax.legend(loc="lower right");
    ax.hlines(y=0, color='k', 
              xmin=curr_dt['human_system_start_time'].min(), 
              xmax=curr_dt['human_system_start_time'].max())


    # sub_dir = plot_dir_base + curr_field_ID.split('_')[3] + "/"
    # print ("curr_dt.county.unique()")
    # print (curr_dt.county.unique()[0])
    # file_name =  sub_dir + curr_dt.county.unique()[0] + "_" + curr_field_ID + ".pdf"
    # os.makedirs(sub_dir, exist_ok=True)
    cc = str(curr_dt.cntrd_ln.unique()[0]) + "_" + str(curr_dt.cntrd_lt.unique()[0])
    file_name =  plot_dir_base + curr_dt.county.unique()[0] + "_" + cc + ".pdf"
    os.makedirs(plot_dir_base, exist_ok=True)

    plt.savefig(fname = file_name, dpi=400, transparent=False, bbox_inches='tight'); # 
    plt.close()


print ("done")
end_time = time.time()
print(end_time - start_time)

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import numpy as np
import pandas as pd

import scipy
import scipy.signal
import os, os.path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sb

from pylab import rcParams

# %%
import shutup
shutup.please()

# %%
# size = 15
# title_FontSize = 10
# legend_FontSize = 8
# tick_FontSize = 12
# label_FontSize = 14

# params = {'legend.fontsize': 'large',
#           'figure.figsize': (6, 4),
#           'axes.labelsize': size,
#           'axes.titlesize': size,
#           'xtick.labelsize': size * 0.75,
#           'ytick.labelsize': size * 0.75,
#           'axes.titlepad': 10}


# #
# #  Once set, you cannot change them, unless restart the notebook
# #
# plt.rc('font', family = 'Palatino')
# plt.rcParams['xtick.bottom'] = False
# plt.rcParams['ytick.left'] = False
# plt.rcParams['xtick.labelbottom'] = False
# plt.rcParams['ytick.labelleft'] = False
# plt.rcParams.update(params)
# # pylab.rcParams.update(params)
# plt.rc('text', usetex=True)

# %%

# %%

# %%
def generate_sine_and_Cosie_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies) + np.cos((3 * np.pi) * frequencies)
    
    return x, y


# %%
SAMPLE_RATE = 44100
DURATION = 3

_, nice_tone_2  = generate_sine_and_Cosie_wave(freq = 2,  sample_rate = SAMPLE_RATE, duration = DURATION)
_, nice_tone_10 = generate_sine_and_Cosie_wave(freq = 10,  sample_rate = SAMPLE_RATE, duration = DURATION)
_, nice_tone_20 = generate_sine_and_Cosie_wave(freq = 20,  sample_rate = SAMPLE_RATE, duration = DURATION)
_, nice_tone_30 = generate_sine_and_Cosie_wave(freq = 30,  sample_rate = SAMPLE_RATE, duration = DURATION)
_, nice_tone_40 = generate_sine_and_Cosie_wave(freq = 40, sample_rate = SAMPLE_RATE, duration = DURATION)


_, noise_tone_100 = generate_sine_and_Cosie_wave(freq = 100, sample_rate = SAMPLE_RATE, duration = DURATION)
_, noise_tone_200 = generate_sine_and_Cosie_wave(freq = 200, sample_rate = SAMPLE_RATE, duration = DURATION)
_, noise_tone_300 = generate_sine_and_Cosie_wave(freq = 300, sample_rate = SAMPLE_RATE, duration = DURATION)
_, noise_tone_400 = generate_sine_and_Cosie_wave(freq = 400, sample_rate = SAMPLE_RATE, duration = DURATION)
_, noise_tone_500 = generate_sine_and_Cosie_wave(freq = 500, sample_rate = SAMPLE_RATE, duration = DURATION)
_, noise_tone_600 = generate_sine_and_Cosie_wave(freq = 600, sample_rate = SAMPLE_RATE, duration = DURATION)

noise_tone_100 = noise_tone_100 * 0.4
noise_tone_200 = noise_tone_200 * 0.4
noise_tone_300 = noise_tone_300 * 0.4
noise_tone_400 = noise_tone_400 * 0.4
noise_tone_500 = noise_tone_500 * 0.4
noise_tone_600 = noise_tone_600 * 0.4

nice_tone = nice_tone_2 + nice_tone_10 + nice_tone_20 + \
            nice_tone_30 + nice_tone_40 

noisy_tone = nice_tone + noise_tone_100 + noise_tone_200 + \
             noise_tone_300 + noise_tone_400 + noise_tone_500 + \
             noise_tone_600

print (len(nice_tone))

# %%
plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/flowchart/"

fig, ax = plt.subplots()
length_=3 # 10
height_=1 # 2
fig.set_size_inches(length_, height_)

#######
ax.plot(noisy_tone, '-', c = "r", linewidth=.7)
ax.plot(nice_tone, '-', c='dodgerblue', linewidth=2) #  

ax.grid(False);
plt.xlim([0, 10000])
plt.axis('off');
file_name = plot_dir + "denoise_" + str(length_) + "_by_" + str (height_)+".pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

file_name = plot_dir + "denoise_" + str(length_) + "_by_" + str (height_)+".png"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %%
with plt.xkcd(scale=1, length=200):
    plt.plot(noisy_tone, '-', c = "dodgerblue")
    plt.plot(nice_tone, '-', c='r') #  linewidth=2,

# %%
import igraph
from igraph import Graph, EdgeSeq


# %%
def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations


# %%
nr_vertices = 15
v_label = list(map(str, range(nr_vertices)))
G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
lay = G.layout('rt')

position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(G) # sequence of edges
E = [e.tuple for e in G.es] # list of edges

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

labels = v_label

# %%

# %%

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=Xe,
                         y=Ye,
                         mode='lines',
                         line=dict(color='rgb(210,210,210)', width=1),
                         hoverinfo='none'
                         ))

fig.add_trace(go.Scatter(x=Xn,
                         y=Yn,
                         mode='markers',
                         name='bla',
                         marker=dict(symbol='circle-dot',
                                     size=18,
                                     color='#6175c1',    #'#DB4551',
                                     line=dict(color='rgb(50,50,50)', width=1)),
                  text=labels,
                  hoverinfo='text',
                  opacity=1
                  ))


# layout = Layout(
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)'
# )

# fig.update_layout(showlegend=False, 
#                   plot_bgcolor='rgba(0,0,0,0)',
#                  # paper_bgcolor='rgba(0,0,0,0)'
#                  )
# fig.update_xaxes(visible=False)
# fig.update_yaxes(visible=False)

axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )


fig.update_layout(# title= 'Tree with Reingold-Tilford Layout',
                  # annotations=make_annotations(position, v_label),
                  font_size=12,
                  showlegend=False,
                  xaxis=axis,
                  yaxis=axis,
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)'
                  )



# %% [markdown]
# # No axis on plots.

# %%
import datetime
from datetime import date, timedelta
import time

import os, os.path
from os import listdir
from os.path import isfile, join

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys


sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
field_IDs=["45478_WSDA_SF_2018", "103932_WSDA_SF_2017"]

# %%
VI_idx = "EVI"
SG_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

SG_data_4_plot=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(SG_data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    SG_data_4_plot=pd.concat([SG_data_4_plot, curr_file])

SG_data_4_plot.reset_index(drop=True, inplace=True)
SG_data_4_plot.head(2)

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

landsat_DF.reset_index(drop=True, inplace=True)

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
for curr_ID in field_IDs:
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
    ax1.grid(False)
    curr_SG_data=SG_data_4_plot[SG_data_4_plot.ID==curr_ID].copy()
    
    curr_landsat_DF=landsat_DF[landsat_DF.ID==curr_ID].copy()
    curr_landsat_DF=nc.add_human_start_time_by_system_start_time(curr_landsat_DF)

    curr_landsat_DF.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_data.sort_values(by=['human_system_start_time'], inplace=True)

    curr_year=curr_SG_data.human_system_start_time.dt.year.unique()[0]
    curr_landsat_DF=curr_landsat_DF[curr_landsat_DF.human_system_start_time.dt.year==curr_year]

    ax1.plot(curr_SG_data['human_system_start_time'], curr_SG_data[VI_idx], 
            linewidth=4, color="dodgerblue", label="smoothed") 

    ax1.scatter(curr_landsat_DF['human_system_start_time'], 
                curr_landsat_DF[VI_idx], 
               s=20, c="r", label="raw")
    

    # ax1.set_ylabel(VI_idx) # , labelpad=20); # fontsize = label_FontSize,
    # ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    # ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    # ax1.legend(loc="upper right");
    ####################################
    # plt.yticks(np.arange(0, 1.05, 0.2));
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    plt.axis('off');
    ax1.set_ylim(-0.1, 1.05);

    file_name = plot_dir + curr_ID + "_" + VI_idx + "_flowChart.pdf"
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);
    
    file_name = plot_dir + curr_ID + "_" + VI_idx + "_flowChart.png"
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

    plt.close('all')

# %% [markdown]
# # NDVI-Ratio Method

# %%
curr_ID="227_WSDA_SF_2016"
given_year = int(curr_ID[-4:])

min_year = pd.to_datetime(datetime.datetime(given_year-1, 1, 1))
max_year = pd.to_datetime(datetime.datetime(given_year+1, 12, 31))

# %%
curr_SG_data=SG_data_4_plot[SG_data_4_plot.ID==curr_ID].copy()

curr_landsat_DF=landsat_DF[landsat_DF.ID==curr_ID].copy()
curr_landsat_DF.drop(["NDVI"], axis=1, inplace=True)
curr_landsat_DF = curr_landsat_DF[curr_landsat_DF["EVI"].notna()]
curr_landsat_DF=nc.add_human_start_time_by_system_start_time(curr_landsat_DF)

# %%
curr_SG_data.sort_values(by=['human_system_start_time'], inplace=True)
curr_SG_data.reset_index(drop=True, inplace=True)

curr_landsat_DF.sort_values(by=['human_system_start_time'], inplace=True)
curr_landsat_DF.reset_index(drop=True, inplace=True)


# %%
def SG_clean_SOS_orchardinPlot_VerticalLine(raw_dt, SG_dt, idx, ax, onset_cut=0.5, offset_cut=0.5):
    assert (len(SG_dt['ID'].unique()) == 1)

    #############################################
    ###
    ###      find SOS's and EOS's
    ###
    #############################################
    ratio_colName = idx + "_ratio"
    SEOS_output_columns = ['ID', idx, 'human_system_start_time', 
                           ratio_colName, 'SOS', 'EOS', 'season_count']
    
    all_poly_and_SEOS = pd.DataFrame(data = None, 
                                     index = np.arange(4*14*len(SG_dt)), 
                                     columns = SEOS_output_columns)
    unique_years = SG_dt['human_system_start_time'].dt.year.unique()
    
    pointer_SEOS_tab = 0
    SG_dt = SG_dt[SEOS_output_columns[0:3]]
    
    """
    detect SOS and EOS in each year
    """
    yr_count = 0
    for yr in unique_years:
        curr_field_yr = SG_dt[SG_dt['human_system_start_time'].dt.year == yr].copy()
        y_orchard = curr_field_yr[curr_field_yr['human_system_start_time'].dt.month >= 5]
        y_orchard = y_orchard[y_orchard['human_system_start_time'].dt.month <= 10]
        y_orchard_range = max(y_orchard[idx]) - min(y_orchard[idx])

        if y_orchard_range > 0.3:
            curr_field_yr = nc.addToDF_SOS_EOS_White(pd_TS = curr_field_yr,
                                                     VegIdx = idx, 
                                                     onset_thresh = onset_cut, 
                                                     offset_thresh = offset_cut)
            curr_field_yr = nc.Null_SOS_EOS_by_DoYDiff(pd_TS=curr_field_yr, min_season_length=40)
        else:
            VegIdx_min = curr_field_yr[idx].min()
            VegIdx_max = curr_field_yr[idx].max()
            VegRange = VegIdx_max - VegIdx_min + sys.float_info.epsilon
            curr_field_yr[ratio_colName] = (curr_field_yr[idx] - VegIdx_min) / VegRange
            curr_field_yr['SOS'] = 666
            curr_field_yr['EOS'] = 666
        #############################################
        ###             plot
        #############################################
        ax.plot(SG_dt['human_system_start_time'], SG_dt[idx], c='dodgerblue', linewidth=3,
                label= 'SG' if yr_count == 0 else "");

        ax.scatter(raw_dt['human_system_start_time'], raw_dt[idx], 
                   s=10, c='r', label="raw" if yr_count == 0 else "");
        ###
        ###   plot SOS and EOS
        SOS = curr_field_yr[curr_field_yr['SOS'] != 0]
        if len(SOS)>0: # dataframe might be empty
            if SOS.iloc[0]['SOS'] != 666:
                ax.scatter(SOS['human_system_start_time'], SOS['SOS'], marker='+', s=155, c='g', 
                          label="")
                # annotate SOS
                for ii in np.arange(0, len(SOS)):
                    style = dict(size=12, color='g', rotation='vertical')
                    ax.text(x=SOS.iloc[ii]['human_system_start_time'].date(), 
                            y=-0.18, 
                            s=str(SOS.iloc[ii]['human_system_start_time'].date())[5:], #
                            **style)
            else:
                 ax.plot(curr_field_yr['human_system_start_time'], 
                         np.ones(len(curr_field_yr['human_system_start_time']))*1, 
                         c='g', linewidth=2);
        #  EOS
        EOS = curr_field_yr[curr_field_yr['EOS'] != 0]
        if len(EOS)>0: # dataframe might be empty
            if EOS.iloc[0]['EOS'] != 666:
                ax.scatter(EOS['human_system_start_time'], EOS['EOS'], 
                           marker='+', s=155, c='r', 
                           label="")

                # annotate EOS
                for ii in np.arange(0, len(EOS)):
                    style = dict(size=12, color='r', rotation='vertical')
                    ax.text(x = EOS.iloc[ii]['human_system_start_time'].date(), 
                            y = -0.18, 
                            s = str(EOS.iloc[ii]['human_system_start_time'].date())[5:],
                            **style)
        yr_count += 1
    
    plt.axhline(y=onset_cut, color='gray', linestyle='-.')

    # ax.set_title(SG_dt['ID'].unique()[0] + ", cut: " + str(onset_cut) + ", " + idx);
    ax.set(ylabel=idx)
    ax.set_xlim([SG_dt.human_system_start_time.min() - timedelta(10), 
                 SG_dt.human_system_start_time.max() + timedelta(10)])
    
    ax.set_ylim([-0.3, 1.15])
    from matplotlib.dates import MonthLocator, DateFormatter
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b'))
    plt.axis('off');
    # ax.legend(loc="upper left");


# %%
fig, ax1 = plt.subplots(1, 1, figsize=(8, 3),
                    sharex='col', sharey='row',
                    gridspec_kw={'hspace': 0.1, 'wspace': .1});

ax1.grid(True);
# ax3.grid(True); ax4.grid(True); ax5.grid(True); ax6.grid(True);

# Plot EVIs
SG_clean_SOS_orchardinPlot_VerticalLine(raw_dt = curr_landsat_DF, SG_dt = curr_SG_data,
                                        idx=VI_idx,
                                        ax=ax1,
                                        onset_cut=0.3, offset_cut=0.3);

file_name = plot_dir + curr_ID + "_" + VI_idx + "_flowChart_NDVIRatio.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + curr_ID + "_" + VI_idx + "_flowChart_NDVIRatio.png"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);


# %%

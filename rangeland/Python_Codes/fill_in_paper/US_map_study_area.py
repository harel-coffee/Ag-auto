# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib


import numpy as np
import matplotlib.colors as mcolors

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
# import missingno as msno
import os
# import wget
# import openpyxl
import math

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"

Min_data_dir_base = data_dir_base + "Min_Data/"
Mike_dir = data_dir_base + "Mike/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

plot_dir = data_dir_base + "00_plots/"

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

county_id_name_fips = abb_dict["county_fips"]
county_id_name_fips.head(2)

# %%
state_abb_state_fips = county_id_name_fips[["state", "state_fips", "EW_meridian"]].copy()
state_abb_state_fips.drop_duplicates(inplace=True)
state_abb_state_fips.reset_index(drop=True, inplace=True)
print(state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %%
abb_dict.keys()

# %%

# %%
state_abb_state_fips["SoI"] = 0
state_abb_state_fips.loc[state_abb_state_fips.state.isin(SoI_abb), "SoI"] = 1
state_abb_state_fips.loc[state_abb_state_fips.EW_meridian=="W", "SoI"] = 2

# remove kentucky
state_abb_state_fips.loc[state_abb_state_fips.state=="KY", "SoI"] = 0

state_abb_state_fips.head(2)

# %%
state_abb_state_fips.SoI.unique()

# %%
gdf = gpd.read_file(data_dir_base + 'cb_2018_us_state_500k.zip')
gdf.head(3)

# %%
gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf.to_crs({'init':'epsg:2163'})

# %%

# %%
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for with Matplotlib for main map
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box from the main map
ax.axis('off')

# create map of all states except AK and HI in the main map axis
visframe[~visframe.state.isin(["AK", "HI"])].plot(color='lightblue', 
                                                  linewidth=0.8, ax=ax, edgecolor='0.8');

# Add Alaska Axis (x, y, width, height)
# akax = fig.add_axes([0.1, 0.17, 0.17, 0.16])   

# Add Hawaii Axis(x, y, width, height)
# hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])

# %%
gdf = gdf.merge(state_abb_state_fips[["state_fips", "EW_meridian", "SoI"]], 
                left_on='STATEFP', right_on='state_fips')

gdf.head(2)

# %%
# remove Puerto Rico 
gdf = gdf[~(gdf.state.isin(["PR", "VI"]))] # "AK", "HI"


# %%
# plot_color_gradients('Sequential',
#                      ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                       'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                       'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

# Apply this to the gdf to ensure all states are assigned colors by the same func
def makeColorColumn(gdf, variable, vmin, vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGnBu)
    gdf['value_determined_color'] = gdf[variable].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))
    return gdf


# %%
# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())

# Choose the continuous colorscale "YlOrBr" from https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = "YlOrBr"

# %%
# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = makeColorColumn(gdf, variable, vmin, vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

hfont = {'fontname':'Helvetica'}

# add a title and annotation
txt_ = "Food Insecurity by Percentage of State Households\n2019-2021"
# ax.set_title(txt_, **hfont, fontdict={'fontsize': '42', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = "States in the study (east and west of meridian)"
cbax.set_title(txt_, **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
# comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
text_ = "Data: USDA Economic Research Service"  
# ax.annotate(text_, xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# create map
# Note: we're going state by state here because of unusual coloring behavior 
# when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# add Alaska
# akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
# akax.axis('off')
# polygon to clip western islands
# polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
# alaska_gdf = gdf[gdf.state=='AK']
# alaska_gdf.clip(polygon).plot(color=gdf[gdf.state=='AK'].value_determined_color, 
#                               linewidth=0.8,ax=akax, edgecolor='0.8')

# add Hawaii
# hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
# hiax.axis('off')
# polygon to clip western islands
# hipolygon = Polygon([(-160,0),(-160,90),(-120,90),(-120,0)])
# hawaii_gdf = gdf[gdf.state=='HI']
# hawaii_gdf.clip(hipolygon).plot(column=variable, color=hawaii_gdf['value_determined_color'], 
#                                 linewidth=0.8,ax=hiax, edgecolor='0.8')

# fig.savefig(os.getcwd()+'study_area.pdf',dpi=400, bbox_inches="tight")
# bbox_inches="tight" keeps the vis from getting cut off at the edges in the saved png

# %%
visframe.value_determined_color.unique()

# %%
len(gdf[gdf.SoI==1].state)

# %%
len(gdf[gdf.SoI==0].state)

# %%
import plotly.express as px

dfa = px.data.election()
dfa.head(2)

# %%
len(SoI)

# %%
# import plotly.express as px

# df = px.data.election()
# geojson = px.data.election_geojson()

# fig = px.choropleth(df, geojson=geojson, color="winner",
#                     locations="district", featureidkey="properties.district",
#                     projection="mercator", hover_data=["Bergeron", "Coderre", "Joly"]
#                    )
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()

# %%
# fig = px.choropleth(gdf, geojson=geojson, color="SoI",
# #                     locations="district", featureidkey="properties.district",
# #                     projection="mercator", hover_data=["Bergeron", "Coderre", "Joly"]
#                    )
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()

# %%

# %%
import plotly.express as px

fig = px.choropleth(locations=list(gdf.state), 
                    locationmode="USA-states", color=np.arange(1, len(list(gdf.state))+1), scope="usa")
fig.show()

# %%
import plotly.graph_objects as go

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

df['text'] = df['state'] + '<br>' + \
    'Beef ' + df['beef'] + ' Dairy ' + df['dairy'] + '<br>' + \
    'Fruits ' + df['total fruits'] + ' Veggies ' + df['total veggies'] + '<br>' + \
    'Wheat ' + df['wheat'] + ' Corn ' + df['corn']

fig = go.Figure(data=go.Choropleth(locations=df['code'],
                                   z=df['total exports'].astype(float),
                                   locationmode='USA-states',
                                   colorscale='Reds',
                                   autocolorscale=False,
                                   text=df['text'], # hover text
                                   marker_line_color='white', # line markers between states
                                   colorbar_title="Millions USD"))

fig.update_layout(title_text='2011 US Agriculture Exports by State<br>(Hover for breakdown)',
                  geo = dict(scope='usa',
                             projection=go.layout.geo.Projection(type = 'albers usa'),
                             showlakes=True, # lakes
                             lakecolor='rgb(255, 255, 255)'))

fig.show()

# %%
df.head(2)

# %%

# %%
fig = go.Figure(data=go.Choropleth(locations = gdf.state,
                                   z = gdf['SoI'].astype(float),
                                   locationmode = 'USA-states',
                                   colorscale = 'Reds',
                                   autocolorscale = False,
                                   text = df['text'], # hover text
                                   marker_line_color = 'white', # line markers between states
                                   colorbar_title = "Millions USD"
                                ))

fig.update_layout(title_text = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
                  geo = dict(
                  scope = 'usa',
                  projection = go.layout.geo.Projection(type = 'albers usa'),
                  showlakes = True, # lakes
                  lakecolor = 'rgb(255, 255, 255)'))

fig.show()

# %%
from matplotlib.colors import ListedColormap

col_dict = {0: "#ffffd9",
            1: "#40b5c4",
            2: "dodgerblue"}

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

# Let's also define the description of 
# each category : 1 (blue) is Sea; 2 (red) is burnt, etc... 
# Order should be respected here ! Or using another dict maybe could help.
labels = np.array(["west meridian", "east meridian", "not of interest"])
len_lab = len(labels)

# prepare normalizer
# Prepare bins for the normalizer
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
print(norm_bins)

# Make normalizer and formatter
norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])


# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
txt_ = "Food Insecurity by Percentage of State Households\n2019-2021"
# ax.set_title(txt_, **hfont, fontdict={'fontsize': '42', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()

# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = "States in the study (east and west of meridian)"
cbax.set_title(txt_, **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# reformat tick labels on legend
sm._A = []
comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
text_ = "Data: USDA Economic Research Service"  
# ax.annotate(text_, xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# create map
# Note: we're going state by state here because of unusual coloring behavior 
# when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = col_dict[gdf[gdf.state==row.state][0:1].SoI.item()]
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

# fig.savefig(os.getcwd()+'study_area.pdf',dpi=400, bbox_inches="tight")
# bbox_inches="tight" keeps the vis from getting cut off at the edges in the saved png

# %%
tick_legend_FontSize = 12

params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.5,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
from matplotlib.colors import ListedColormap
east_color = "#40b5c4"
col_dict = {2: "dodgerblue",
            # 0: "#ffffd9", 
            0: "#C0C0C0",
            1: east_color}

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box around the vis
hfont = {'fontname':'Helvetica'} # set the font for the visualization to Helvetica

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = col_dict[gdf[gdf.state==row.state][0:1].SoI.item()]
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

import matplotlib.patches as mpatches
legend_dict = {"west meridian": "dodgerblue",
               "east meridian": east_color}
patchList = []
for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

plt.legend(handles=patchList);
fig.savefig(plot_dir + 'study_area_python.pdf', dpi=800, bbox_inches="tight")

# %%

# %%
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# "#ffffd9" # "#40b5c4" # east meridian
east_color = "#FE420F"
col_dict = {2: "dodgerblue",
            # 0: "#ffffd9", 
#             0: "#929591",
            0: "#C0C0C0",
            1: east_color
           }

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box around the vis
hfont = {'fontname':'Helvetica'} # set the font for the visualization to Helvetica

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = col_dict[gdf[gdf.state==row.state][0:1].SoI.item()]
        # print (c)
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

import matplotlib.patches as mpatches

legend_dict = {"west meridian": "dodgerblue",
               "east meridian": east_color}
patchList = []
for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key, lw=0.5)
        patchList.append(data_key)

plt.legend(handles=patchList);
fig.savefig(plot_dir + 'study_area_python_red.pdf', dpi=800, bbox_inches="tight")

# %%

# %%
print (state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %%
print (len(SoI_abb))
SoI_abb[1:3]

# %%
state_abb_state_fips_SoI = state_abb_state_fips[state_abb_state_fips.state.isin(SoI_abb)].copy()

# %%
len(state_abb_state_fips_SoI[state_abb_state_fips_SoI.EW_meridian=="W"])

# %%
len(state_abb_state_fips_SoI[state_abb_state_fips_SoI.EW_meridian=="E"])

# %%
tonsor_states = "Alabama, Arkansas, California, Colorado, Florida, Georgia, Idaho," + \
                 "Illinois, Iowa, Kansas, Kentucky," + \
                 "Louisiana, Missouri, Mississippi, Montana," + \
                 "Nebraska, New Mexico, North Dakota, Oklahoma, Oregon," + \
                 "South Dakota, Tennessee, Texas, Virginia, Wyoming"
tonsor_states = tonsor_states.replace(", ", ",")
tonsor_states = tonsor_states.split(",")
print (len(tonsor_states))
tonsor_states[:3]

# %%
abb_dict.keys()

# %%
SoI_full_names = [x for x in SoI if not (x in tonsor_states)]
SoI_full_names

# %%
tonsor_states

# %%
SoI

# %%
shannon_regions_dict_abbr = {"region_1_region_2" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI', 'NY', 'NJ'], 
                             "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
                             "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                             "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
                             "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
                             "region_7" : ['IA', 'KS', 'MO', 'NE'],
                             "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                             "region_9" : ['AZ', 'CA', 'HI', 'NV'],
                             "region_10": ['AK', 'ID', 'OR', 'WA']}

col_dict = {"region_1_region_2": "cyan",
            "region_3": "black", 
            "region_4": "green",
            "region_5": "tomato",
            "region_6": "red",
            "region_7": "dodgerblue",
            "region_8": "dimgray", # gray: "#C0C0C0"
            "region_9": "#ffd343", # mild yellow
            "region_10": "steelblue"}

# %%
import matplotlib.patches as mpatches

# We create a colormar from our list of colors
cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
# **************************
# set the value column that will be visualised
variable = "SoI"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

visframe["region"] = "NA"

# some states may not be in a region
# those will be gray
visframe["region_color"] = "#C0C0C0" 

for a_key in shannon_regions_dict_abbr.keys():
    visframe.loc[visframe["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key
    visframe.loc[visframe["state"].isin(shannon_regions_dict_abbr[a_key]), 'region_color'] = col_dict[a_key]
visframe.head(2)

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box around the vis
hfont = {'fontname':'Helvetica'} # set the font for the visualization to Helvetica

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = vf["region_color"].item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

############### add Alaska and Hawaii
# add Alaska
akax = fig.add_axes([0.1, 0.17, 0.2, 0.19])   
akax.axis('off')
# polygon to clip western islands
polygon = Polygon([(-170,50),(-170,72),(-140, 72),(-140,50)])
alaska_gdf = gdf[gdf.state=='AK'] # I donno why visframe is not working here
alaska_gdf.clip(polygon).plot(color = visframe[visframe.state=="AK"]["region_color"].item(), 
                              linewidth=0.8, ax=akax, edgecolor='0.8')


# add Hawaii
# visframe = gdf.to_crs({'init':'epsg:2163'})
hiax = fig.add_axes([.28, 0.20, 0.1, 0.1])   
hiax.axis('off')
# polygon to clip western islands
hipolygon = Polygon([(-160,0),(-160,90), (-120,90), (-120,0)])
hawaii_gdf = gdf[gdf.state=='HI'] # I donno why visframe is not working here
hawaii_gdf.clip(hipolygon).plot(color = visframe[visframe.state=="HI"]["region_color"].item(), 
                                 linewidth=0.8, ax=hiax, edgecolor='0.8')


legend_dict = col_dict
patchList = []
for key in legend_dict:
    if key== "region_1_region_2":
        label_ = "region 1 & 2"
    else:
        label_ = key.replace("_", " ")
    data_key = mpatches.Patch(color=legend_dict[key], label=label_, lw=0.2)
    patchList.append(data_key)

plt.legend(handles=patchList, bbox_to_anchor=(5, 1), ncols=2,  fontsize=12);
fig.savefig(plot_dir + 'regions.pdf', dpi=800, bbox_inches="tight")

# %%

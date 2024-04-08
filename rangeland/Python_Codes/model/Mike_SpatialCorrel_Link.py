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

# %% [markdown]
# # First download the data
#
# https://geographicdata.science/book/data/airbnb/regression_cleaning.html

# %%
# # !pip3 install geopandas
# # !pip3 install geopy
# # !pip3 install mapclassify
# # !pip3 install googlemaps

# %%
# %matplotlib inline

import numpy as np
import requests
import pandas as pd
import geopandas as gpd
#import googlemaps
from scipy.spatial.distance import cdist

# %%
import os
os.chdir("/Users/hn/Documents/01_research_data/RangeLand/Data/Mike_SpatialCorr/SanDiego/")
          

# %%
# outdated and do not exist.
# download directly from https://insideairbnb.com/get-the-data/
# or correct the old dates (2016-07-07) to what currently is there: 2023-12-04

# url = 'http://data.insideairbnb.com/united-states/'\
#       'ca/san-diego/2016-07-07/data/'\
#       'listings.csv.gz'
# r = requests.get(url)
# with open('listings.csv.gz', 'wb') as fo:
#     fo.write(r.content)

# url = 'http://data.insideairbnb.com/united-states/'\
#       'ca/san-diego/2016-07-07/data/'\
#       'calendar.csv.gz'
# r = requests.get(url)
# with open('calendar.csv.gz', 'wb') as fo:
#     fo.write(r.content)


# url = 'http://data.insideairbnb.com/united-states/'\
#       'ca/san-diego/2016-07-07/visualisations/'\
#       'neighbourhoods.geojson'
# r = requests.get(url)
# with open('neighbourhoods.geojson', 'wb') as fo:
#     fo.write(r.content)

# %%
# url = 'http://data.insideairbnb.com/united-states/'\
#       'ca/san-diego/2016-07-07/data/'\
#       'listings.csv.gz'
# r = requests.get(url)
# with open('listings.csv.gz', 'wb') as fo:
#     fo.write(r.content)

# %%
# url = 'http://data.insideairbnb.com/united-states/'\
#       'ca/san-diego/2023-12-04/data/'\
#       'listings.csv.gz'
# r = requests.get(url)
# with open('listings.csv.gz', 'wb') as fo:
#     fo.write(r.content)

# %% [markdown]
# # Variable set up
#
# ## Parse price

# %%
# import gzip
# with gzip.open('listings.csv.gz', 'rb') as fio:
#     df = pd.read_csv(fio)

# %%
os.getcwd()

# %%
lst = pd.read_csv('listings.csv.gz')
lst['priceN'] = lst['price'].apply(
                    lambda x: float(str(x)\
                                    .replace(',', '')\
                                    .strip('$')))
lst['l_price'] = np.log(lst['priceN'])

# %%
lst.columns

# %%
from shapely.geometry import Point
xys = lst[['longitude', 'latitude']]\
        .apply(lambda row: Point(*row), axis=1)
gdb = gpd.GeoDataFrame(lst.assign(geometry=xys),
                       crs="+init=epsg:4326")

# %% [markdown]
# # Variables
# ## pool

# %%
import re

ams = []
gdb['pool'] = 0
for i in range(gdb.shape[0]):
    r = gdb.loc[i, 'amenities']
    pcs = r.strip('{').strip('}').split(',')
    ams.extend(pcs)
    if re.findall("pool", r.lower()):
        gdb.loc[i, 'pool'] = 1
set(ams)

# %%
## Distance to Balboa park
import geopy

# %%
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Hossein")
bp = geolocator.geocode("Balboa Park, San Diego, US")

# %%
b_ll = bp.longitude, bp.latitude
b_ll

# %%
# Then calculate distance to the park from each house:


# USA Contiguous Albers Equal Area (m.)
# http://epsg.io/102003
tgt_crs = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 "\
          "+lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"

b_xy = gpd.GeoSeries(Point(b_ll), crs=gdb.crs).to_crs(tgt_crs)[0]
b_xy = (b_xy.x, b_xy.y)
# Calculate distance in Km.
d2b = lambda pt: cdist([(pt.x, pt.y)], [b_xy])[0][0] / 1000

# %%
# # !pip3 install mapclassify

import mapclassify
gdb['d2balboa'] = gdb['geometry'].to_crs(tgt_crs).apply(d2b)
gdb.plot(column='d2balboa', scheme='quantiles', k=9, cmap='viridis_r', s=1)

# %%
# Elevation:
# key = open('../google_maps_key').readline().strip('\n')
# I go tthe following key from: https://developers.google.com/maps/documentation/embed/get-api-key

# # !pip3 install googlemaps
import googlemaps
gmaps = googlemaps.Client(key="AIzaSyDrD8vsMZPof9F9YYqPFMPtgtOU_PsHTZg")

# %%
# Google takes lat/lon instead of lon/lat
gmaps.elevation([b_ll[::-1]])

# %%
pts = gdb['geometry'].apply(lambda pt: (pt.y, pt.x))
# %time ele = gmaps.elevation(pts.head().tolist())
ele

# %%
extract_ele = lambda x: pd.Series(x)['elevation']
eleS = pd.Series(ele).apply(extract_ele)
eleS

# %%
# Coastal neighborhood?

# %%
coastal_neighborhoods = ['Wooded Area', 'Ocean Beach', 'Pacific Beach', \
                         'La Jolla', 'Torrey Pines', 'Del Mar Heighs', \
                         'Mission Bay']
def coastal(neigh):
    if neigh in coastal_neighborhoods:
        return 1
    else:
        return 0
gdb['coastal_neig'] = gdb['neighbourhood_cleansed'].apply(coastal)

gdb.plot(column='coastal_neig', s=1,
         categorical=True, legend=True);

# %%
# Large neighborhood

# %%
lrg_nei = gdb.groupby('neighbourhood_cleansed').size() > 25
gdb['lrg_nei'] = gdb['neighbourhood_cleansed'].map(lrg_nei)

# %%
# List to keep
xs = ['accommodates', 'bathrooms', 'bedrooms', 
      'beds', 'neighbourhood_cleansed', 'pool',
      'd2balboa', 'coastal_neig', 'lrg_nei',
      'priceN', 'l_price',
      'geometry', 'id']

# %%
# Dummies

## Room type

# %%

# %%
rt = pd.get_dummies(gdb['room_type'], prefix='rt').rename(columns=lambda x: x.replace(' ', '_'))


# %%
# Property type
def simplify(p):
    bigs = ['House', 'Apartment', 'Condominium', 'Townhouse']
    if p in bigs:
        return p
    else:
        return 'Other'

gdb['property_group'] = gdb['property_type'].apply(simplify)
pg = pd.get_dummies(gdb['property_group'], prefix='pg')

# %%
gdb[['lrg_nei']].info()

# %%
lrg_nei

# %%
# # ! rm 'regression_db.geojson'
final = gdb[xs].join(pg)\
               .join(rt)\
               .rename(columns={'priceN': 'price'})\
               .loc[gdb['lrg_nei']==True, :]\
               .drop(['lrg_nei'], axis=1)\
#                .dropna()

# %%
final = final.rename(columns=dict(neighbourhood_cleansed='neighborhood', 
                          coastal_neig='coastal',
                          l_price = 'log_price'))

# %%
# !rm regression_db.geojson
final.to_file('regression_db.geojson', driver='GeoJSON')
final.info()

# %% [markdown]
# # Following
# https://geographicdata.science/book/notebooks/11_regression.html

# %%

# %%

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# The link Mike sent about [spatial correlation](https://geographicdata.science/book/notebooks/11_regression.html)
#
#
# **First download the data**
#
# https://geographicdata.science/book/data/airbnb/regression_cleaning.html

# %%
import shutup
shutup.please()

# %%
# In the context of this chapter, it makes sense to start with spreg, 
# as that is the only library that will allow us to move into 
# explicitly spatial econometric models
# # !pip3 install contextily
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily

from scipy.stats import ttest_ind

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

# ttest_ind(coastal, not_coastal)

# %% [markdown]
# #### Fit OLS model with Spreg
#
# \begin{equation}
# m1 = spreg.OLS(db[["log_price"]].values, #Dependent variable
#                     db[variable_names].values, # Independent variables
#                name_y="log_price", # Dependent variable name
#                name_x=variable_names # Independent variable name
#                )
# \end{equation}

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
# set(ams)

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

# %% [markdown]
# ### Large neighborhood

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

# %% [markdown]
# # Dummies
#
# ## Room type

# %%
rt = pd.get_dummies(gdb['room_type'], prefix='rt').rename(columns=lambda x: x.replace(' ', '_'))

# %%
rt["rt_Hotel_room"]


# %%

# %%
# Property type
def original_simplify(p):
    """
    This is from the link above. and does not work here.
    I donno what was the data they were working with orginally.
    So, I have to change it.
    """
    bigs = ['House', 'Apartment', 'Condominium', 'Townhouse']
    if p in bigs:
        return p
    else:
        return 'Other'
    
def simplify(p):
    if ("townhouse" in p.lower()):
        return "Townhouse" 
    elif ("apartment" in p.lower()):
        return "Apartment"
    elif ("condo" in p.lower()):
        return "Condo"
    elif ("house" in p.lower()): # we put townhouse first since house is substring of townhouse
        return "House"
    else:
        return 'Other'

gdb['property_group'] = gdb['property_type'].apply(simplify)
pg = pd.get_dummies(gdb['property_group'], prefix='pg')

# %%
gdb['property_type'].unique()

# %%
gdb['property_group'].unique()

# %%

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
[x for x in final.columns if "pg_" in x]

# %%

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
db = geopandas.read_file("regression_db.geojson")

# %%
variable_names = [
    "accommodates",  # Number of people it accommodates
    # "bathrooms",  # Number of bathrooms: all mising from my data.
    # "bedrooms",  # Number of bedrooms : my data has only 5 intances. others are missing.
    "beds",  # Number of beds
    # Below are binary variables, 1 True, 0 False
    "rt_Private_room",  # Room type: private room
    "rt_Shared_room",  # Room type: shared room
    "pg_Condo", # "pg_Condominium",  # Property group: condo
    "pg_House",  # Property group: house
    "pg_Other",  # Property group: other
    "pg_Townhouse",  # Property group: townhouse
]

# %%
db.drop(columns=["bathrooms", "bedrooms"], inplace=True)
db.dropna(inplace=True)

# %%
# db_original = db.copy()

# %%
# db = db[variable_names + ["log_price"]].copy()
# db.head(2)

# %%

# %%
db["rt_Private_room"] = db["rt_Private_room"].astype(float)
db["rt_Shared_room"] = db["rt_Shared_room"].astype(float)
db["pg_Condo"] = db["pg_Condo"].astype(float)
db["pg_House"] = db["pg_House"].astype(float)
db["pg_Other"] = db["pg_Other"].astype(float)
db["pg_Townhouse"] = db["pg_Townhouse"].astype(float)

# %%
db.head(2)

# %%
db["pg_Apartment"].unique()

# %%
m1 = spreg.OLS(db[["log_price"]].values, # Dependent variable
               db[variable_names].values, # Independent variables
               name_y="log_price", # Dependent variable name
               name_x=variable_names, # Independent variable name
              )

print(m1.summary)

# %%
db["pg_Apartment"].unique()

# %%
# Create a Boolean (True/False) with whether a
# property is coastal or not
is_coastal = db.coastal.astype(bool)

# Split residuals (m1.u) between coastal and not
coastal = m1.u[is_coastal]
not_coastal = m1.u[~is_coastal]

# Create histogram of the distribution of coastal residuals
plt.hist(coastal, density=True, label="Coastal")

# Create histogram of the distribution of non-coastal residuals
plt.hist(not_coastal, histtype="step", density=True, linewidth=4, label="Not Coastal",);
plt.vlines(0, 0, 1, linestyle=":", color="k", linewidth=4)
plt.legend();
plt.show();

# %%
from scipy.stats import ttest_ind

ttest_ind(coastal, not_coastal)

# %%
# Create column with residual values from m1
db["residual"] = m1.u

# Obtain the median value of residuals in each neighborhood
medians = (
    db.groupby("neighborhood")
    .residual.median()
    .to_frame("hood_residual")
)


seaborn.set(font_scale=1.25) # Increase fontsize
f = plt.figure(figsize=(15, 3)) # Set up figure
ax = plt.gca() # Grab figure's axis
# Generate bloxplot of values by neighborhood
# Note the data includes the median values merged on-the-fly
seaborn.boxplot(x="neighborhood",
                y="residual",
                ax=ax,
                data=db.merge(medians, how="left", left_on="neighborhood", right_index=True
                             ).sort_values("hood_residual"),
                palette="bwr")

f.autofmt_xdate(rotation=-90) # Rotate the X labels for legibility
plt.show()

# %%
db["pg_Apartment"].unique()

# %%
knn = weights.KNN.from_dataframe(db, k=1)

# %%
lag_residual = weights.spatial_lag.lag_spatial(knn, m1.u)
ax = seaborn.regplot(x=m1.u.flatten(),
                     y=lag_residual.flatten(),
                     line_kws=dict(color="orangered"),
                     ci=None)

ax.set_xlabel("Model Residuals - $u$")
ax.set_ylabel("Spatial Lag of Model Residuals - $W u$");

# %%
# %%time
knn.reweight(k=20, inplace=True) # Re-weight W to 20 nearest neighbors
knn.transform = "R" # Row standardize weights

# Run LISA on residuals
outliers = esda.moran.Moran_Local(m1.u, knn, permutations=9999)


error_clusters = outliers.q % 2 == 1 # Select only LISA cluster cores

# Filter out non-significant clusters
error_clusters &= outliers.p_sim <= 0.001

# Add `error_clusters` and `local_I` columns
ax = (db.assign(error_clusters=error_clusters,
                local_I=outliers.Is
                # Retain error clusters only
               )
      .query("error_clusters" # Sort by I value to largest plot on top
            )
      .sort_values("local_I" # Plot I values
                  )
      .plot("local_I", cmap="bwr", marker="."))

contextily.add_basemap(ax, crs=db.crs) # Add basemap
ax.set_axis_off();

# %%
ax = db.plot("d2balboa", marker=".", s=5)
contextily.add_basemap(ax, crs=db.crs)
ax.set_axis_off();

# %%
balboa_names = variable_names + ["d2balboa"]

# %%
db["pg_Apartment"].unique()

# %%
m2 = spreg.OLS(db[["log_price"]].values,
               db[balboa_names].values,
               name_y="log_price",
               name_x=balboa_names)

# %%
pd.DataFrame([[m1.r2, m1.ar2], [m2.r2, m2.ar2]],
                 index=["M1", "M2"],
                 columns=["R2", "Adj. R2"])

# %%
# Set up table of regression coefficients
pd.DataFrame({# Pull out regression coefficients and
              # flatten as they are returned as Nx1 array
              "Coeff.": m2.betas.flatten(),
              "Std. Error": m2.std_err.flatten(), # Pull out and flatten standard errors
              "P-Value": [i[1] for i in m2.t_stat]}, # Pull out P-values from t-stat object
              index=m2.name_x)

# %%
lag_residual = weights.spatial_lag.lag_spatial(knn, m2.u)
ax = seaborn.regplot(x=m2.u.flatten(),
                     y=lag_residual.flatten(),
                     line_kws=dict(color="orangered"),
                     ci=None)
ax.set_xlabel("Residuals ($u$)")
ax.set_ylabel("Spatial lag of residuals ($Wu$)");

# %%
import statsmodels.formula.api as sm

# %%
f = ("log_price ~ " + " + ".join(variable_names) + " + neighborhood - 1")
print(f)

# %%
m3 = sm.ols(f, data=db).fit()

# %%
# Store variable names for all the spatial fixed effects
sfe_names = [i for i in m3.params.index if "neighborhood[" in i]

pd.DataFrame({"Coef.": m3.params[sfe_names],
              "Std. Error": m3.bse[sfe_names],
              "P-Value": m3.pvalues[sfe_names]})

# %%
db["pg_Apartment"].unique()

# %%
# spreg spatial fixed effect implementation
m4 = spreg.OLS_Regimes(db[["log_price"]].values, # Dependent variable
                       db[variable_names].values, # Independent variables
                       db["neighborhood"].tolist(), # Variable specifying neighborhood membership
                       constant_regi="many", # Allow the constant term to vary by group/regime
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       cols2regi=[False] * len(variable_names),
                       
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y="log_price", # Dependent variable name
                       name_x=variable_names, # Independent variables names
                      )

# %%
np.round(m4.betas.flatten() - m3.params.values, decimals=12)

# %%
db["pg_Apartment"].unique()

# %%
neighborhood_effects = m3.params.filter(like="neighborhood")
neighborhood_effects.head()

# %%
# Create a sequence with the variable names without
# `neighborhood[` and `]`
stripped = neighborhood_effects.index.str.strip("neighborhood[").str.strip("]")
neighborhood_effects.index = stripped # Reindex the neighborhood_effects Series on clean names
neighborhood_effects = neighborhood_effects.to_frame("fixed_effect")
neighborhood_effects.head()

# %%
sd_path = "neighbourhoods.geojson"
neighborhoods = geopandas.read_file(sd_path)

# %%
db["pg_Apartment"].unique()

# %%
# Plot base layer with all neighborhoods in grey
ax = neighborhoods.plot(color="k", linewidth=0, alpha=0.5, figsize=(12, 6))

# Merge SFE estimates (note not every polygon
# receives an estimate since not every polygon
# contains Airbnb properties)
neighborhoods.merge(neighborhood_effects,
                    how="left",
                    left_on="neighbourhood",
                    right_index=True).dropna(subset=["fixed_effect"]
                                            ).plot("fixed_effect",  # Variable to display
                                                   scheme="quantiles",  # Choropleth scheme
                                                   k=7,  # No. of classes in the choropleth
                                                   linewidth=0.1,  # Polygon border width
                                                   cmap="viridis",  # Color scheme
                                                   ax=ax,  # Axis to draw on
                                                  )
# Add basemap
contextily.add_basemap(ax, crs=neighborhoods.crs, source=contextily.providers.CartoDB.PositronNoLabels)
ax.set_axis_off()
plt.show()

# %% [markdown]
# # Spatial regimes

# %%
# Pysal spatial regimes implementation
m5 = spreg.OLS_Regimes(db[["log_price"]].values, # Dependent variable
                       db[variable_names].values, # Independent variables
                       db["coastal"].tolist(), # Variable specifying neighborhood membership
                       constant_regi="many", # Allow the constant term to vary by group/regime
                       
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y="log_price", # Dependent variable name
                       name_x=variable_names, # Independent variables names
                      )

# %%
db["pg_Apartment"].unique()

# %%
# Results table
res = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                    "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                    "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                   },
                   index=m5.name_x)
# Coastal regime
## Extract variables for the coastal regime
coastal = [i for i in res.index if "1_" in i]

## Subset results to coastal and remove the 1_ underscore
coastal = res.loc[coastal, :].rename(lambda i: i.replace("1_", ""))
## Build multi-index column names
coastal.columns = pd.MultiIndex.from_product([["Coastal"], coastal.columns])

# Non-coastal model
## Extract variables for the non-coastal regime
ncoastal = [i for i in res.index if "0_" in i]

## Subset results to non-coastal and remove the 0_ underscore
ncoastal = res.loc[ncoastal, :].rename(lambda i: i.replace("0_", ""))

## Build multi-index column names
ncoastal.columns = pd.MultiIndex.from_product([["Non-coastal"], ncoastal.columns])
pd.concat([coastal, ncoastal], axis=1)

# %%
db["pg_Apartment"].unique()

# %%
m5.chow.joint

# %%
pd.DataFrame(m5.chow.regi, # Chow results by variable
             index=m5.name_x_r, # Name of variables
             columns=["Statistic", "P-value"])

# %% [markdown]
# # Exogenous effects: The SLX model

# %%
# Select only columns in `db` containing the keyword `pg_`
wx = db.filter(like="pg_").apply(lambda y: weights.spatial_lag.lag_spatial(knn, y))\
                              .rename(columns=lambda c: "w_" + c)
wx.drop("w_pg_Apartment", axis=1)

# %%
db["pg_Apartment"].unique()

# %%
# Merge original variables with the spatial lags in `wx`
slx_exog = db[variable_names].join(wx)


m6 = spreg.OLS(db[["log_price"]].values, # Dependent variable
               slx_exog.values, # Independent variables
               name_y="l_price", # Dependent variable name
               name_x=slx_exog.columns.tolist(), # Independent variables names
              )

# %%
db["pg_Apartment"].unique()

# %%
# Collect names of variables of interest
vars_of_interest = (db[variable_names].filter(like="pg_").join(wx).columns)
# Build full table of regression coefficients
pd.DataFrame({"Coeff.": m6.betas.flatten(),
              "Std. Error": m6.std_err.flatten(), # Pull out and flatten standard errors
              "P-Value": [i[1] for i in m6.t_stat], # Pull out P-values from t-stat object
             },
             index=m6.name_x).reindex(vars_of_interest).round(4)

# %%
m5.predy[:3]

# %%
db["pg_Apartment"].unique()

# %%

# %%
# Print values for third observation for columns spanning
# from `pg_Apartment` to `pg_Townhouse`
db.loc[2, "pg_Apartment":"pg_Townhouse"]

# %%
db.pg_Apartment = db.pg_Apartment.astype(float) # not in the tutorial
db_scenario = db.copy()
# Make Apartment 0 and condo 1 for third observation
db_scenario.loc[2, ["pg_Apartment", "pg_Condo"]] = [0, 1]

db_scenario.head(4)

# %%
db["pg_Apartment"].unique()

# %%
db_scenario.loc[2, "pg_Apartment":"pg_Townhouse"]

# %%
db_scenario["pg_Apartment"].unique()

# %%
db_scenario.head(2)

# %%
# Select only columns in `db_scenario` containing the keyword `pg_`
wx_scenario = db_scenario.filter(like="pg")\
                     .apply(lambda y: weights.spatial_lag.lag_spatial(knn, y))

wx_scenario.rename(columns=lambda c: "w_" + c, inplace=True)
wx_scenario.drop("w_pg_Apartment", axis=1)

# %%
db_scenario.head(2)

# %%

# %%
slx_exog_scenario = db_scenario[variable_names].join(wx_scenario)

# Compute new set of predicted values
y_pred_scenario = m6.betas[0] + slx_exog_scenario @ m6.betas[1:]

# %%
print(knn.neighbors[2])

# %%
# Difference between original and new predicted values
(y_pred_scenario - m6.predy).loc[[2] + knn.neighbors[2]]

# %% [markdown]
# # Spatial error

# %%
# Fit spatial error model with `spreg`
# (GMM estimation allowing for heteroskedasticity)
m7 = spreg.GM_Error_Het(db[["log_price"]].values, # Dependent variable
                        db[variable_names].values, # Independent variables
                        w=knn, # Spatial weights matrix
                        name_y="log_price", # Dependent variable name
                        name_x=variable_names, # Independent variables names
                       )

# %%
# Build full table of regression coefficients
pd.DataFrame({"Coeff.": m7.betas.flatten(),
              "Std. Error": m7.std_err.flatten(), # Pull out and flatten standard errors
              "P-Value": [i[1] for i in m7.z_stat], # Pull out P-values from t-stat object
             }, index=m7.name_x).reindex(["lambda"]).round(4) # Subset for lambda parameter


# %% [markdown]
# # Spatial lag

# %%
# Fit spatial lag model with `spreg`
# (GMM estimation)
m8 = spreg.GM_Lag(db[["log_price"]].values, # Dependent variable
                  db[variable_names].values, # Independent variables
                  w=knn, # Spatial weights matrix
                  name_y="log_price", # Dependent variable name
                  name_x=variable_names, # Independent variables names
                 )

# %%
# Build full table of regression coefficients
pd.DataFrame({"Coeff.": m8.betas.flatten(),
              "Std. Error": m8.std_err.flatten(), # Pull out and flatten standard errors
              "P-Value": [i[1] for i in m8.z_stat], # Pull out P-values from t-stat object
             }, index=m8.name_z).round(4)

# %%
# %%time

n_simulations = 100
f, ax = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=True)

ax[0].hist(coastal, color=["r"] * 3, alpha=0.5, density=True, bins=30, label="Coastal", cumulative=True)
ax[1].hist(not_coastal, color="b", alpha=0.5, density=True, bins=30, label="Not Coastal", cumulative=True)
for simulation in range(n_simulations):
    shuffled_residuals = m1.u[np.random.permutation(m1.n)]
    random_coast, random_notcoast = (shuffled_residuals[is_coastal], shuffled_residuals[~is_coastal])
    if simulation == 0:
        label = "Simulations"
    else:
        label = None
    ax[0].hist(random_coast, density=True, histtype="step", color="k", alpha=0.05, bins=30, 
               label=label, cumulative=True)
    ax[1].hist( random_coast, density=True, histtype="step", color="k", alpha=0.05, bins=30, 
               label=label, cumulative=True)
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# # The K-neighbor correlogram

# %%
correlations = []
nulls = []
for order in range(1, 51, 5):
    knn.reweight(k=order, inplace=True) # operates in place, quickly and efficiently avoiding copies
    knn.transform = "r"
    lag_residual = weights.spatial_lag.lag_spatial(knn, m1.u)
    random_residual = m1.u[np.random.permutation(len(m1.u))]
    # identical to random neighbors in KNN
    random_lag_residual = weights.spatial_lag.lag_spatial(knn, random_residual)
    correlations.append(np.corrcoef(m1.u.flatten(), lag_residual.flatten())[0, 1])
    nulls.append(np.corrcoef(m1.u.flatten(), random_lag_residual.flatten())[0, 1])

# %%
plt.plot(range(1, 51, 5), correlations)
plt.plot(range(1, 51, 5), nulls, color="orangered")
plt.hlines(np.mean(correlations[-3:]), *plt.xlim(), linestyle=":", color="k")
plt.hlines(np.mean(nulls[-3:]), *plt.xlim(), linestyle=":", color="k")
plt.text(s="Long-Run Correlation: ${:.2f}$".format(np.mean(correlations[-3:])), x=25, y=0.3)
plt.text(s="Long-Run Null: ${:.2f}$".format(np.mean(nulls[-3:])), x=25, y=0.05)
plt.xlabel("$K$: number of nearest neighbors")
plt.ylabel("Correlation between site \n and neighborhood average of size $K$")
plt.show()

# %%

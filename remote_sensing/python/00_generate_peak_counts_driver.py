


###
### load libraries
###
import cProfile
import random
import numpy.random as rand
import warnings
import scipy.io as sio
import os
import sys
import argparse


import numpy as np
import pandas as pd
import geopandas as gpd
import sys
from IPython.display import Image
from shapely.geometry import Point, Polygon
from math import factorial
import datetime
import time
import scipy

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb


import sys
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')
import remote_sensing_core as remote_core

"""
# Plan for rough estimate:
   1. perennials, and grasses and non irrigated (Why google slide doess not say annuals)
"""

################################################################
#####
#####                   directories
#####
################################################################
data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/01_NDVI_TS/Grant/"


################################################################
#####
#####                   Data Reading
#####
################################################################
# 
# See how you can list all files in the given directory to produce 
# file_names 
#
file_names = ["Grant_2018_TS.csv"]
file_N = file_names[0]

a_df = pd.read_csv(data_dir + file_N)
a_df = initial_clean(a_df)
a_df.head(2)

# Count distict values, use nunique:
pprint (a_df['geo'].nunique())

# Count only non-null values, use count:
print (a_df['geo'].count())

# Count total values including null values, use size attribute:
print (a_df['geo'].size)

################################################################
#####
#####             List of unique polygons
#####
################################################################
polygon_list = a_df['geo'].unique()


################################################################
#####
#####         Iterate through polygons and count peaks
#####
################################################################
output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                  'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                  'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                  'peak_Doy', 'peak_value']

all_polygons_and_their_peaks = pd.DataFrame(data=None, 
                                            columns=output_columns)
for a_poly in polygon_list:
    a_field = a_df[a_df['geo']==a_poly]
    
    ### 
    ###  There is a chance that a polygon is repeated twice?
    ###
    
    X = a_field['doy']
    y = a_field['NDVI']
    freedom_df = 7
    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    
    # Generate spline basis with "freedom_df" degrees of freedom
    x_basis = cr(X, df=freedom_df, constraints='center')

    # Fit model to the data
    model = LinearRegression().fit(x_basis, y)

    # Get estimates
    y_hat = model.predict(x_basis)
    
    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    # peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
    # max_peaks = peaks_LWLS_1[0]
    # peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

    peaks_spline = peakdetect(y_hat, lookahead = 10, delta=0)
    max_peaks = peaks_spline[0]
    peaks_spline = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)
    
    DoYs_series = pd.Series(int(peaks_spline[0]))
    peaks_series = pd.Series(peaks_spline[1])
    
    peak_df = pd.DataFrame({ 
                       'peak_Doy': pd.Series(int(peaks_spline[0])), 
                       'peak_value': pd.Series(peaks_spline[1]) 
                      }) 

    
    WSDA_df = keep_WSDA_columns(a_field)
    WSDA_df = WSDA_df.drop_duplicates()
    
    WSDA_df = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index()
    # WSDA_df = pd.concat([WSDA_df, peak_df], axis=1, ignore_index=True)
    WSDA_df = WSDA_df.join(peak_df)
    all_polygons_and_their_peaks = all_polygons_and_their_peaks.append(WSDA_df, sort=False)
    
    """
    # first I decided to add all DoY and peaks in one row to avoid
    # multiple rows per (field, year)
    # However, in this way, each pair of (field, year)
    # can have different column sizes.
    # So, we cannot have one dataframe to include everything in it.
    # so, we will have to do dictionary to save out puts.
    # Lets just do replicates... easier to handle perhaps down the road.
    #
    DoY_colNames = [i + j for i, j in zip(\
                                          ["DoY_"]*(len(DoYs_series)+1), \
                                          [str(i) for i in range(1, len(DoYs_series)+1)] )] 
   
    peak_colNames = [i + j for i, j in zip(\
                                           ["peak_"]*(len(peaks_series)+1), \
                                           [str(i) for i in range(1, len(peaks_series)+1)] )]
    
    WSDA_df[DoY_colNames] = pd.DataFrame([DoYs_series], index=WSDA_df.index)
    WSDA_df[peak_colNames] = pd.DataFrame([peaks_series], index=WSDA_df.index)
    """




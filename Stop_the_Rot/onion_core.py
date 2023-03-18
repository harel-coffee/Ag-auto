import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
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

import os, os.path
import math
import sys


###########################################################


def relative_humidity_simple(a_DT_RH):
    """
    https://earthscience.stackexchange.com/questions/16570/
    how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    
    𝑅𝐻 ≈ 100 −5 ( 𝑇 − 𝑇𝐷) where T is temp and TD is Dew point.
    """
    RH = 100 - (5 * (a_DT_RH["Avg_Temp_C"] - a_DT_RH["Dew_Point_C"]))
    return RH


def relative_humidity_complex_1(a_DT_RH):
    """
    https://earthscience.stackexchange.com/questions/16570/
    how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure
    """
    # a_DT_RH["simple_RH"] = RL
    b = 17.625
    c = 243.04

    numerator = (b*c) * (a_DT_RH["Dew_Point_C"] - a_DT_RH["Avg_Temp_C"])
    denim = (c + a_DT_RH["Dew_Point_C"]) * ( c + a_DT_RH["Avg_Temp_C"])

    exponent = numerator / denim
    exponent = exponent.values
    RH = [100*math.exp(thing) for thing in exponent]
    return (RH)

###########################################################

def read_xlsx_with_only_1sheet(path_n_fileName, head_count=1):
    """
    Here I am assuming the xlsx file has only one sheet.
    Since the Date column is not sorted I read and sort
    """
    A = pd.read_excel(io = path_n_fileName, header=head_count) # , sheet_name = 'Real Time Soil Moisture data'

    A.sort_values(by = 'Date', inplace=True)
    A.reset_index(drop= True, inplace=True)
    return(A)

###########################################################

def minMax_standardize_soilMoisture(a_df):
    
    a_df.Sensor1 = (a_df.Sensor1 - min(a_df.Sensor1)) / (max(a_df.Sensor1) - min(a_df.Sensor1))
    a_df.Sensor2 = (a_df.Sensor2 - min(a_df.Sensor2)) / (max(a_df.Sensor2) - min(a_df.Sensor2))
    a_df.Sensor3 = (a_df.Sensor3 - min(a_df.Sensor3)) / (max(a_df.Sensor3) - min(a_df.Sensor3))
    a_df.Sensor4 = (a_df.Sensor4 - min(a_df.Sensor4)) / (max(a_df.Sensor4) - min(a_df.Sensor4))
    return(a_df)

def clean_ABC_Weather_Toss2ndTable(ABC_df_F):
    """
    ABC_df_F is a dataframe
    """
    ABC_df = ABC_df_F.copy()

    # Detect the first row where all row is NaN
    first_NaN_row = ABC_df[ABC_df.isnull().any(axis=1)].index[0]

    # Toss everything after the first row of NaNs (i.e. the second table in sheets)
    ABC_df = ABC_df.iloc[0:first_NaN_row]
    return(ABC_df)

def read_ABC_weather_XLSfile(an_add_fileName, header_rows = 1, skip_Rows = 3):
    """

    an_add_fileName is the full address and name of the file to read
    This function lists all the sheets in the excel file, and reads them,
    concatenates them, and returns a dataframe.

    Here we are assuming all sheets have consistent structure similar to 
    "CLEANED (3) Station ABC Weather Data 2017.xlsx" and nothing weird will happen later.

    """
    xl = pd.ExcelFile(an_add_fileName)
    EX_sheet_names = xl.sheet_names  # see all sheet names

    outDF = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    for sheet in EX_sheet_names:
        curr_sheet = pd.read_excel(io = an_add_fileName, header = header_rows, sheet_name = sheet, skiprows = skip_Rows)
        curr_sheet = clean_ABC_Weather_Toss2ndTable(curr_sheet)
        outDF = pd.concat([outDF, curr_sheet])
    outDF.reset_index(drop=True, inplace=True)
    return (outDF)


def convert_to_numerictype(data_F, given_cols):
    for col in given_cols:
        data_F[col] = pd.to_numeric(data_F[col], errors='coerce')

def group_by_compute_stats(a_dataF, stat, stat_columns, group_by_cols):
    #
    # subset the needed columns
    #
    needed_columns = stat_columns + group_by_cols
    a_dataF_sub = a_dataF[needed_columns].copy()

    #
    # find stats per group
    #
    if stat == "mean":
        
        a_dataF_sub = a_dataF_sub.groupby(by = group_by_cols).mean()

        # change the column names so we know what we have done

        postfix = ["_averages"] * len(stat_columns)
        new_column_names = [i + j for i, j in zip(stat_columns, postfix)]
        a_dataF_sub.columns = new_column_names

        # reset index converts the index back to a column and adds it to proper rows
        a_dataF_sub = a_dataF_sub.reset_index()

    if stat == "sum":
        a_dataF_sub = a_dataF_sub.groupby(by = group_by_cols).sum()

        # change the column names so we know what we have done

        postfix = ["_cumulative_sum"] * len(stat_columns)
        new_column_names = [i + j for i, j in zip(stat_columns, postfix)]
        a_dataF_sub.columns = new_column_names

        # reset index converts the index back to a column and adds it to proper rows
        a_dataF_sub = a_dataF_sub.reset_index()
    return(a_dataF_sub)












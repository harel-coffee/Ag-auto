####
#### Nov. 29, 2021
####
"""
 The goal is to cluster the fields. In this driver
 all fields are in 1 big fat bag; i.e. county and crop type does not matter.
 As heat/spring moves along the earth, the same crop will have different
 temporal signature. Interannual variability is there too. The same crop, same location,
 can have different signature in different years. 

 I do not think there is a point to this actually, but, here we are!
"""

import csv
import numpy as np
import pandas as pd
import datetime
import time
import os, os.path
# from patsy import cr

import sys
start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters                   
###
####################################################################################
indeks = sys.argv[1]

"""
 There must be 36 data points per year if interal size is 10.
 But, for extra security lets say we have only 35 since
 the regularization step depends on 1st and last image dates;
 i.e. we did not do Jan-1 to Jan-10, Jan-11 to Jan-20 and so on.
 We found the last date and first date, and we made bins based on the date range!
 So, there might be a mismatch between the fields
"""
data_per_year = 35

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
SF_data_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"
data_dir = "/data/hydro/users/Hossein/NASA/05_SG_TS/"
clusters_dir = "/data/hydro/users/Hossein/NASA/07_clusters/"

output_dir = clusters_dir
os.makedirs(output_dir, exist_ok=True)
print ("_________________________________________________________")
print ("data dir is: " + data_dir)
print ("_________________________________________________________")
print ("output_dir is: " + output_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################
county_names = ["SG_AdamBenton2016", "SG_FranklinYakima2018", 
                "SG_Grant2017", "SG_Monterey2014", "SG_Walla2015"]

x_harmonized_df = pd.DataFrame(data = None)

for county in county_names:
    SG_df = pd.read_csv(data_dir + county + "_" + indeks + "_JFD.csv")
    SG_df['human_system_start_time'] = pd.to_datetime(SG_df['human_system_start_time'])
    SG_df["ID"] = SG_df["ID"].astype(str) # Monterays ID will be read as integer, convert to string

    # pick up only 1 year worth of data!
    SG_df = SG_df[SG_df['human_system_start_time'].dt.year == int(county[-4:])]
    print (SG_df['human_system_start_time'].dt.year.unique())

    """
       Harmonize the x-values; i.e. make all of them to have length data_per_year.
    """
    fields_IDs = SG_df.ID.unique()
    for an_ID in fields_IDs:
        field = SG_df[SG_df.ID == an_ID]
        if field.shape[0] > data_per_year:
            SG_df.drop(field.index[data_per_year:], inplace=True)

    x_harmonized_df = pd.concat([x_harmonized_df, SG_df])

x_harmonized_df.reset_index(drop=True, inplace=True)
out_name = output_dir + "x_harmonized_" + indeks + "_JFD.csv"
x_harmonized_df.to_csv(out_name, index = False)

print ("done")
end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


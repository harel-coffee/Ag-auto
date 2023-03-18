####
#### May 23, 2022
####


import csv
import numpy as np
import pandas as pd
from math import factorial
import scipy
import scipy.signal
import os, os.path
from datetime import date
import datetime
import time
import sys
start_time = time.time()

# search path for modules# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

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

indeks = sys.argv[1] # red, blue, NIR, green, short_I_1, short_I_2

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print ("Terminal Arguments are: ")
print (indeks)
print ("__________________________________________")

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"

output_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/wide_DFs/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   band names
###
########################################################################################
moreBand_fNames=[x for x in os.listdir(data_dir) if x.endswith(".csv")]
moreBand_fNames=[x for x in moreBand_fNames if "moreBands" in x]
moreBand_fNames=[x for x in moreBand_fNames if indeks in x]

print ("moreBand_fNames")
print (moreBand_fNames)

########################################################################################
###
###                   process data
###
########################################################################################
aBand_allCounties_wide_DF = pd.DataFrame()

wide_colnames = [indeks + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID"] + wide_colnames

for a_file in moreBand_fNames:
    a_file_df = pd.read_csv(data_dir + a_file)
    a_file_df = a_file_df.round({indeks: 2})
    a_county_wide = pd.DataFrame(columns=columnNames,
                                 index=range(len(a_file_df.ID.unique())))
    a_county_wide["ID"] = a_file_df.ID.unique()

    for an_ID in a_file_df.ID.unique():
        curr_df = a_file_df[a_file_df.ID==an_ID]
        a_county_wide_indx = a_county_wide[a_county_wide.ID==an_ID].index
        left_col=indeks + "_1"
        right_col=indeks+"_36"
        a_county_wide.loc[a_county_wide_indx, left_col:right_col] = curr_df[indeks].values[:36]
    
    aBand_allCounties_wide_DF=pd.concat([aBand_allCounties_wide_DF, a_county_wide])

####################################################################################
###
###                   Write the outputs
###
####################################################################################
# aBand_allCounties_wide_DF.drop_duplicates(inplace=True)
# aBand_allCounties_wide_DF.dropna(inplace=True)

out_name = output_dir + "wide_allCounties_" + indeks + ".csv"
aBand_allCounties_wide_DF.to_csv(out_name, index = False)


end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


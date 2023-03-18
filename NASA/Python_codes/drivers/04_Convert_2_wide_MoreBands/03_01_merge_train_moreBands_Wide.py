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

# sys.path.append('/home/hnoorazar/NASA/')
# import NASA_core as nc
# import NASA_plot_core as ncp

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/wide_DFs/"

output_dir = data_dir
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                    band names
###
########################################################################################
moreBand_fNames=["wide_allCounties_NIR.csv", "wide_allCounties_blue.csv",
                 "wide_allCounties_green.csv", "wide_allCounties_red.csv",
                 "wide_allCounties_short_I_1.csv", "wide_allCounties_short_I_2.csv"]

########################################################################################
###
###                   process data
###
########################################################################################
allsBand_allCounties_wide_DF = pd.read_csv(data_dir + moreBand_fNames[0])

for a_file in moreBand_fNames[1:]:
    a_file_df = pd.read_csv(data_dir + a_file)
    allsBand_allCounties_wide_DF = pd.merge(allsBand_allCounties_wide_DF, a_file_df, on=['ID'], how='left')

####################################################################################
###
###                   Write the outputs
###
####################################################################################
# aBand_allCounties_wide_DF.drop_duplicates(inplace=True)
# aBand_allCounties_wide_DF.dropna(inplace=True)

out_name = output_dir + "wide_allCounties_moreBands.csv"
allsBand_allCounties_wide_DF.to_csv(out_name, index = False)


end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


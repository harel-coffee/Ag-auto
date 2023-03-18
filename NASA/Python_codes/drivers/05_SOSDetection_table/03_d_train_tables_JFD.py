####
#### last update Nov. 29, 2021
####
"""
  This is not efficient. I am doing the following two scenarios in a
  sequential fashion.
   a. three filters: irrigated fields, NASS out, survey date correct
   b. one filter: irrigated fields

   The least could be done is do it in parallel fashion!
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
county = sys.argv[1]
indeks = sys.argv[2]
SEOS_cut = sys.argv[3]
"""
  White SOS and EOS params
    SEOS_cut will be provided as integers. 3, 4, 5, 35.
    Convert to 0.3, 0.4, 0.4, or 0.35
"""
if len(SEOS_cut) == 1:
    onset_cut = int(SEOS_cut) / 10.0
elif len(SEOS_cut) == 2:
    onset_cut = int(SEOS_cut) / 100.0

offset_cut = onset_cut
print("SEOS_cut is {}.".format(SEOS_cut))
print("onset_cut is {} and offset_cut is {}.".format(onset_cut, offset_cut))

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
SF_data_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"
data_dir = "/data/hydro/users/Hossein/NASA/05_SG_TS/"
SOS_table_dir = "/data/hydro/users/Hossein/NASA/06_SOS_tables/"

output_dir = SOS_table_dir # + str(int(onset_cut*10)) + "_EOS" + str(int(offset_cut*10)) + "/"
os.makedirs(output_dir, exist_ok=True)
print ("_________________________________________________________")
print ("data dir is: " + data_dir)
print ("_________________________________________________________")
print ("output_dir is: " +  output_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################
SG_df = pd.read_csv(data_dir + "SG_" + county + "_" + indeks + "_JFD.csv")
SG_df['human_system_start_time'] = pd.to_datetime(SG_df['human_system_start_time'])
SG_df["ID"] = SG_df["ID"].astype(str) # Monterays ID will be read as integer, convert to string

"""
  train data should be correct. Therefore, we need to filter by
  last survey year, toss out NASS, and we are sticking to irrigated
  fields for now.
"""
SF_data = pd.read_csv(SF_data_dir + county + ".csv")
SF_data["ID"] = SF_data["ID"].astype(str)

if county == "Monterey2014":
    SF_data['Crop2014'] = SF_data['Crop2014'].str.lower().str.replace(" ", "_").str.replace(",", "").str.replace("/", "_")
else:
    SF_data['CropTyp'] = SF_data['CropTyp'].str.lower().str.replace(" ", "_").str.replace(",", "").str.replace("/", "_")

if county != "Monterey2014":
    # filter by last survey date. Last 4 digits of county name!
    print("No. of fields in SF_data is {}.".format(len(SF_data.ID.unique())))
    SF_data = nc.filter_by_lastSurvey(SF_data, year = county[-4:])
    print("No. of fields in SF_data after survey year is {}.".format(len(SF_data.ID.unique())))
    SF_data = nc.filter_out_NASS(SF_data)         # Toss NASS
    print("No. of fields in SF_data after NASS is {}.".format(len(SF_data.ID.unique())))
    SF_data = nc.filter_out_nonIrrigated(SF_data) # keep only irrigated lands
    print("No. of fields in SF_data after Irrigation is {}.".format(len(SF_data.ID.unique())))

    fk = list(SF_data.ID)
    SG_df = SG_df[SG_df.ID.isin(fk)]

SG_df = pd.merge(SG_df, SF_data, on=['ID'], how='left')

print ("columns of SG_df right after merging is: ")
print (SG_df.head(2))

####################################################################################
###
###                   process data
###
####################################################################################
SG_df.reset_index(drop=True, inplace=True)
SG_df = nc.initial_clean(df=SG_df, column_to_be_cleaned = indeks)

### List of unique polygons
polygon_list = SG_df['ID'].unique()

print ("_________________________________________________________")
print("polygon_list is of length {}.".format(len(polygon_list)))

# 
# 25 columns
#
ratio_name = indeks + "_ratio"
SEOS_output_columns = ['ID', 'human_system_start_time', indeks,
                       ratio_name, 'SOS', 'EOS', 'season_count']

#
# The reason I am multiplying len(SG_df) by 4 is that we can have at least two
# seasons which means 2 SOS and 2 EOS. So, at least 4 rows are needed in case of double-cropped.
#
min_year = SG_df.human_system_start_time.dt.year.min()
max_year = SG_df.human_system_start_time.dt.year.max()
no_years = max_year - min_year + 1
all_poly_and_SEOS = pd.DataFrame(data = None, 
                                 index = np.arange(4*no_years*len(SG_df)), 
                                 columns = SEOS_output_columns)
counter = 0
pointer_SEOS_tab = 0
###########
###########  Re-order columns of the read data table to be consistent with the output columns
###########
SG_df = SG_df[SEOS_output_columns[0:3]]

print ("line 144")

for a_poly in polygon_list:
    if (counter % 1000 == 0):
        print ("_________________________________________________________")
        print ("counter: " + str(counter))
        print (a_poly)
    curr_field = SG_df[SG_df['ID']==a_poly].copy()
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    # extract unique years. Should be 2008 thru 2021.
    # unique_years = list(pd.DatetimeIndex(curr_field['human_system_start_time']).year.unique())
    unique_years = curr_field['human_system_start_time'].dt.year.unique()
    
    """
    detect SOS and EOS in each year
    """
    for yr in unique_years:
        curr_field_yr = curr_field[curr_field['human_system_start_time'].dt.year == yr].copy()

        # Orchards EVI was between more than 0.3
        y_orchard = curr_field_yr[curr_field_yr['human_system_start_time'].dt.month >= 5]
        y_orchard = y_orchard[y_orchard['human_system_start_time'].dt.month <= 10]
        y_orchard_range = max(y_orchard[indeks]) - min(y_orchard[indeks])

        if y_orchard_range > 0.3:
            #######################################################################
            ###
            ###             find SOS and EOS, and add them to the table
            ###
            #######################################################################
            curr_field_yr = nc.addToDF_SOS_EOS_White(pd_TS = curr_field_yr, 
                                                     VegIdx = indeks, 
                                                     onset_thresh = onset_cut, 
                                                     offset_thresh = offset_cut)

            ##
            ##  Kill false detected seasons 
            ##
            curr_field_yr = nc.Null_SOS_EOS_by_DoYDiff(pd_TS=curr_field_yr, min_season_length=40)

            #
            # extract the SOS and EOS rows 
            #
            SEOS = curr_field_yr[(curr_field_yr['SOS'] != 0) | curr_field_yr['EOS'] != 0]
            SEOS = SEOS.copy()
            # SEOS = SEOS.reset_index() # not needed really
            SOS_tb = curr_field_yr[curr_field_yr['SOS'] != 0]
            if len(SOS_tb) >= 2:
                SEOS["season_count"] = len(SOS_tb)
                # re-order columns of SEOS so they match!!!
                SEOS = SEOS[all_poly_and_SEOS.columns]
                all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
                pointer_SEOS_tab += len(SEOS)
            else:
                # re-order columns of fine_granular_table so they match!!!
                curr_field_yr["season_count"] = 1
                curr_field_yr = curr_field_yr[all_poly_and_SEOS.columns]

                aaa = curr_field_yr.iloc[0].values.reshape(1, len(curr_field_yr.iloc[0]))
                
                all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                pointer_SEOS_tab += 1
        else: 
            """
             here are potentially apples, cherries, etc.
             we did not add EVI_ratio, SOS, and EOS. So, we are missing these
             columns in the data frame. So, use 666 as proxy
            """
            aaa = np.append(curr_field_yr.iloc[0], [666, 666, 666, 1])
            aaa = aaa.reshape(1, len(aaa))
            all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
            pointer_SEOS_tab += 1

        counter += 1

####################################################################################
###
###                   Write the outputs
###
####################################################################################
# replace the following line with dropna.
# all_poly_and_SEOS = all_poly_and_SEOS[0:(pointer_SEOS_tab)]
all_poly_and_SEOS.dropna(inplace=True)

out_name = output_dir + "SC_train_" + county + "_"  + indeks + str(SEOS_cut) + "_irr_NoNASS_SurvCorrect_JFD.csv"
all_poly_and_SEOS.to_csv(out_name, index = False)



####################################################################################
###
###                   Do it again with only one filter: Irrigated fields; i.e.
###                   Forget about NASS and last survey date filter
###
####################################################################################


####################################################################################
###
###                   Read data
###
####################################################################################
SG_df = pd.read_csv(data_dir + "SG_" + county + "_" + indeks + "_JFD.csv")
SG_df['human_system_start_time'] = pd.to_datetime(SG_df['human_system_start_time'])
SG_df["ID"] = SG_df["ID"].astype(str) # Monterays ID will be read as integer, convert to string

SF_data = pd.read_csv(SF_data_dir + county + ".csv")
SF_data["ID"] = SF_data["ID"].astype(str)

if county == "Monterey2014":
    SF_data['Crop2014'] = SF_data['Crop2014'].str.lower().str.replace(" ", "_").str.replace(",", "").str.replace("/", "_")
else:
    SF_data['CropTyp'] = SF_data['CropTyp'].str.lower().str.replace(" ", "_").str.replace(",", "").str.replace("/", "_")

if county != "Monterey2014":
    SF_data = nc.filter_out_nonIrrigated(SF_data) # keep only irrigated lands
    print("No. of fields in SF_data after Irrigation is {}.".format(len(SF_data.ID.unique())))

    fk = list(SF_data.ID)
    SG_df = SG_df[SG_df.ID.isin(fk)]

SG_df = pd.merge(SG_df, SF_data, on=['ID'], how='left')

print ("columns of SG_df right after merging is: ")
print (SG_df.head(2))

####################################################################################
###
###                   process data
###
####################################################################################
SG_df.reset_index(drop=True, inplace=True)
SG_df = nc.initial_clean(df=SG_df, column_to_be_cleaned = indeks)

### List of unique polygons
polygon_list = SG_df['ID'].unique()

print ("_________________________________________________________")
print("polygon_list is of length {}.".format(len(polygon_list)))

# 
# 25 columns
#
ratio_name = indeks + "_ratio"
SEOS_output_columns = ['ID', 'human_system_start_time', indeks,
                       ratio_name, 'SOS', 'EOS', 'season_count']

#
# The reason I am multiplying len(SG_df) by 4 is that we can have at least two
# seasons which means 2 SOS and 2 EOS. So, at least 4 rows are needed in case of double-cropped.
#
min_year = SG_df.human_system_start_time.dt.year.min()
max_year = SG_df.human_system_start_time.dt.year.max()
no_years = max_year - min_year + 1
all_poly_and_SEOS = pd.DataFrame(data = None, 
                                 index = np.arange(4*no_years*len(SG_df)), 
                                 columns = SEOS_output_columns)
counter = 0
pointer_SEOS_tab = 0
###########
###########  Re-order columns of the read data table to be consistent with the output columns
###########
SG_df = SG_df[SEOS_output_columns[0:3]]

print ("line 144")

for a_poly in polygon_list:
    if (counter % 1000 == 0):
        print ("_________________________________________________________")
        print ("counter: " + str(counter))
        print (a_poly)
    curr_field = SG_df[SG_df['ID']==a_poly].copy()
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    # extract unique years. Should be 2008 thru 2021.
    # unique_years = list(pd.DatetimeIndex(curr_field['human_system_start_time']).year.unique())
    unique_years = curr_field['human_system_start_time'].dt.year.unique()
    
    """
    detect SOS and EOS in each year
    """
    for yr in unique_years:
        curr_field_yr = curr_field[curr_field['human_system_start_time'].dt.year == yr].copy()

        # Orchards EVI was between more than 0.3
        y_orchard = curr_field_yr[curr_field_yr['human_system_start_time'].dt.month >= 5]
        y_orchard = y_orchard[y_orchard['human_system_start_time'].dt.month <= 10]
        y_orchard_range = max(y_orchard[indeks]) - min(y_orchard[indeks])

        if y_orchard_range > 0.3:
            #######################################################################
            ###
            ###             find SOS and EOS, and add them to the table
            ###
            #######################################################################
            curr_field_yr = nc.addToDF_SOS_EOS_White(pd_TS = curr_field_yr, 
                                                     VegIdx = indeks, 
                                                     onset_thresh = onset_cut, 
                                                     offset_thresh = offset_cut)

            ##
            ##  Kill false detected seasons 
            ##
            curr_field_yr = nc.Null_SOS_EOS_by_DoYDiff(pd_TS=curr_field_yr, min_season_length=40)

            #
            # extract the SOS and EOS rows 
            #
            SEOS = curr_field_yr[(curr_field_yr['SOS'] != 0) | curr_field_yr['EOS'] != 0]
            SEOS = SEOS.copy()
            # SEOS = SEOS.reset_index() # not needed really
            SOS_tb = curr_field_yr[curr_field_yr['SOS'] != 0]
            if len(SOS_tb) >= 2:
                SEOS["season_count"] = len(SOS_tb)
                # re-order columns of SEOS so they match!!!
                SEOS = SEOS[all_poly_and_SEOS.columns]
                all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
                pointer_SEOS_tab += len(SEOS)
            else:
                # re-order columns of fine_granular_table so they match!!!
                curr_field_yr["season_count"] = 1
                curr_field_yr = curr_field_yr[all_poly_and_SEOS.columns]
                aaa = curr_field_yr.iloc[0].values.reshape(1, len(curr_field_yr.iloc[0]))
                all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                pointer_SEOS_tab += 1
        else: 
            """
             here are potentially apples, cherries, etc.
             we did not add EVI_ratio, SOS, and EOS. So, we are missing these
             columns in the data frame. So, use 666 as proxy
            """
            aaa = np.append(curr_field_yr.iloc[0], [666, 666, 666, 1])
            aaa = aaa.reshape(1, len(aaa))
            all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
            pointer_SEOS_tab += 1

        counter += 1

####################################################################################
###
###                   Write the outputs
###
####################################################################################
# replace the following line with dropna.
# all_poly_and_SEOS = all_poly_and_SEOS[0:(pointer_SEOS_tab)]
all_poly_and_SEOS.dropna(inplace=True)

out_name = output_dir + "SC_train_" + county + "_"  + indeks + str(SEOS_cut) + "_irrOneFilter_JFD.csv"
all_poly_and_SEOS.to_csv(out_name, index = False)


print ("done")
end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))




# ---
# jupyter:
#   jupytext:
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
import pandas as pd
import numpy as np
import os, sys
from datetime import date, datetime

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis

import scipy, scipy.signal
# import pickle, h5py

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc


# %%
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d

# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img


# %%
VI_TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/09_WSU_students2022Survey/"

# %%
IDcolName = "ID"
file_names = ["L7_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv",
              "L8_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv"]

# %%
VI_TS_df = pd.DataFrame()
for f_ in file_names:
    df = pd.read_csv(VI_TS_dir+f_)
    VI_TS_df = pd.concat([VI_TS_df, df])
    
del(df, f_)

# %%
VI_TS_df.rename(columns={"GlobalID": "ID"}, inplace=True)

IDs = VI_TS_df.ID.unique()

# %%
EVI_TS_df = VI_TS_df.copy()
NDVI_TS_df = VI_TS_df.copy()

EVI_TS_df  = EVI_TS_df[['ID', 'EVI', 'system_start_time']]
NDVI_TS_df = NDVI_TS_df[['ID', 'NDVI', 'system_start_time']]

# %%
print (NDVI_TS_df.shape)
NDVI_TS_df.head(2)

# %%
print (EVI_TS_df.shape)
EVI_TS_df.head(2)

# %%
EVI_TS_df.dropna(subset=["EVI"], inplace=True)
NDVI_TS_df.dropna(subset=["NDVI"], inplace=True)

EVI_TS_df.sort_values(by=["ID", "system_start_time"], inplace=True)
NDVI_TS_df.sort_values(by=["ID", "system_start_time"], inplace=True)

EVI_TS_df.reset_index(drop=True, inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)

# %%
EVI_TS_df  = nc.add_human_start_time_by_system_start_time(EVI_TS_df)
NDVI_TS_df = nc.add_human_start_time_by_system_start_time(NDVI_TS_df)

# %%
print (NDVI_TS_df.shape)
NDVI_TS_df.head(2)

# %%
print (EVI_TS_df.shape)
EVI_TS_df.head(2)

# %%
EVI_TS_df = nc.initial_clean(df = EVI_TS_df, column_to_be_cleaned = "EVI")
NDVI_TS_df = nc.initial_clean(df = NDVI_TS_df, column_to_be_cleaned = "NDVI")

# %%
print (EVI_TS_df.shape)
EVI_TS_df.head(2)

# %%
print (NDVI_TS_df.shape)
NDVI_TS_df.head(2)

# %% [markdown]
# # Remove Outliers

# %%
EVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
EVI_TS_df.reset_index(drop=True, inplace=True)

NDVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)

# %%
EVI_noOutlier = pd.DataFrame(data = None,
                         index = np.arange(EVI_TS_df.shape[0]), 
                         columns = EVI_TS_df.columns)
indeks = "EVI"
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = EVI_TS_df[EVI_TS_df[IDcolName]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        #************************************************
        #
        #    Set negative indeks values to zero.
        #
        #************************************************
        """
         we are killing some of the ourliers here and put them
         in the normal range! do we want to do it here? No, lets do it later.
        """
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0 
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0 

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """ 
        if len(no_Outlier_TS) > 0:
            EVI_noOutlier[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]
            
EVI_noOutlier.drop_duplicates(inplace=True)
print (EVI_noOutlier.shape)
EVI_noOutlier.head(2)

# %%
NDVI_noOutlier = pd.DataFrame(data = None,
                         index = np.arange(NDVI_TS_df.shape[0]), 
                         columns = NDVI_TS_df.columns)
indeks = "NDVI"
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = NDVI_TS_df[NDVI_TS_df[IDcolName]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        #************************************************
        #
        #    Set negative indeks values to zero.
        #
        #************************************************
        """
         we are killing some of the ourliers here and put them
         in the normal range! do we want to do it here? No, lets do it later.
        """
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0 
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0 

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """ 
        if len(no_Outlier_TS) > 0:
            NDVI_noOutlier[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

NDVI_noOutlier.drop_duplicates(inplace=True)
print (NDVI_noOutlier.shape)
NDVI_noOutlier.head(2)

# %%
NDVI_TS_df = NDVI_noOutlier.copy()
EVI_TS_df = EVI_noOutlier.copy()

del(NDVI_noOutlier, EVI_noOutlier)

# %%
print (NDVI_TS_df.shape)
NDVI_TS_df.head(2)

# %%
print (EVI_TS_df.shape)
EVI_TS_df.head(2)

# %% [markdown]
# # Remove Jumps

# %%
EVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
EVI_TS_df.reset_index(drop=True, inplace=True)

NDVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)

# %%
EVI_noJump = pd.DataFrame(data = None,
                          index = np.arange(EVI_TS_df.shape[0]), 
                          columns = EVI_TS_df.columns)
indeks = "EVI"
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = EVI_TS_df[EVI_TS_df[IDcolName]==a_poly].copy()
    
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field, 
                                                        give_col = indeks, 
                                                        maxjump_perDay = 0.018)

    EVI_noJump[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

del(indeks)
EVI_noJump.drop_duplicates(inplace=True)
print (f"{EVI_noJump.shape=}")
EVI_noJump.head(2)

# %%
NDVI_noJump = pd.DataFrame(data = None,
                          index = np.arange(NDVI_TS_df.shape[0]), 
                          columns = NDVI_TS_df.columns)
indeks = "NDVI"
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = NDVI_TS_df[NDVI_TS_df[IDcolName]==a_poly].copy()
    
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field, 
                                                        give_col = indeks, 
                                                        maxjump_perDay = 0.018)

    NDVI_noJump[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

del(indeks)
NDVI_noJump.drop_duplicates(inplace=True)
print (f"{NDVI_noJump.shape=}")
NDVI_noJump.head(2)

# %%
NDVI_TS_df = NDVI_noJump.copy()
EVI_TS_df = EVI_noJump.copy()

del(EVI_noJump, NDVI_noJump)

# %%
NDVI_TS_df.head(2)

# %%
EVI_TS_df.head(2)

# %%
EVI_TS_df.drop(["system_start_time"], axis=1, inplace=True)
NDVI_TS_df.drop(["system_start_time"], axis=1, inplace=True)

EVI_TS_df['human_system_start_time'] = pd.to_datetime(EVI_TS_df['human_system_start_time'])
NDVI_TS_df['human_system_start_time'] = pd.to_datetime(NDVI_TS_df['human_system_start_time'])

# %% [markdown]
# # Regularize

# %%
EVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
EVI_TS_df.reset_index(drop=True, inplace=True)

NDVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)

# %%
indeks = "EVI"
regular_window_size = 10

reg_cols = ['ID', 'human_system_start_time', indeks]

st_yr = EVI_TS_df.human_system_start_time.dt.year.min()
end_yr = EVI_TS_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
reg_EVI = pd.DataFrame(data = None,
                       index = np.arange(nrows), 
                       columns = reg_cols)

counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = EVI_TS_df[EVI_TS_df[IDcolName]==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = indeks, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
    if (counter == 0):
        print ("reg_EVI columns:",     reg_EVI.columns)
        print ("regularized_TS.columns", regularized_TS.columns)    
    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    reg_EVI[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

reg_EVI.drop_duplicates(inplace=True)
reg_EVI.dropna(inplace=True)

del(indeks)
reg_EVI.drop_duplicates(inplace=True)
print (f"{reg_EVI.shape=}")
reg_EVI.head(2)

# %%
indeks = "NDVI"
regular_window_size = 10

reg_cols = ['ID', 'human_system_start_time', indeks]

st_yr = NDVI_TS_df.human_system_start_time.dt.year.min()
end_yr = NDVI_TS_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
reg_NDVI = pd.DataFrame(data = None,
                       index = np.arange(nrows), 
                       columns = reg_cols)
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = NDVI_TS_df[NDVI_TS_df[IDcolName]==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = indeks, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
    if (counter == 0):
        print ("reg_NDVI columns:",     reg_NDVI.columns)
        print ("regularized_TS.columns", regularized_TS.columns)    
    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    reg_NDVI[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

reg_NDVI.drop_duplicates(inplace=True)
reg_NDVI.dropna(inplace=True)

del(indeks)
reg_NDVI.drop_duplicates(inplace=True)
print (f"{reg_NDVI.shape=}")
reg_NDVI.head(2)

# %%
NDVI_TS_df = reg_NDVI.copy()
EVI_TS_df = reg_EVI.copy()

del(reg_EVI, reg_NDVI)

# %%
NDVI_TS_df.head(2)

# %%
EVI_TS_df.head(2)

# %%
out_name = VI_TS_dir + "EVI_regular_WSUStudentSurvey2022.csv"
EVI_TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
EVI_TS_df.to_csv(out_name, index = False)

out_name = VI_TS_dir + "NDVI_regular_WSUStudentSurvey2022.csv"
NDVI_TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
NDVI_TS_df.to_csv(out_name, index = False)

# %% [markdown]
# #### HouseKeeping

# %%
del(no_Outlier_TS, regularized_TS, st_yr, end_yr, no_days, 
    reg_cols, row_pointer, a_poly, out_name, nrows, no_steps, 
    regular_window_size, curr_field)

# %% [markdown]
# # SG

# %%
EVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
EVI_TS_df.reset_index(drop=True, inplace=True)

NDVI_TS_df.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)


# %%
counter = 0
indeks = "EVI"
for a_poly in IDs:
    if (counter % 500 == 0):
        print (f"{counter=}")

    curr_field = EVI_TS_df[EVI_TS_df[IDcolName]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[indeks].values, window_length=7, polyorder=3)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    if counter == 0:
        print(curr_field.head(2))
        print ("==================================================================================")
        print(f"{curr_field.index[0:5]=}")
        print ("==================================================================================")
        # print (EVI_TS_df.loc[curr_field.index, ])
        # print ("==================================================================================")
        print(f"{len(SG)=}")
        print ("==================================================================================")
        print (SG[1:10])

    EVI_TS_df.loc[curr_field.index, indeks] = SG
    counter += 1

EVI_TS_df.drop_duplicates(inplace=True)
EVI_TS_df.dropna(inplace=True)

del(indeks, curr_field, a_poly)
print ()
print ("=============== done ===============")
print (f"{EVI_TS_df.shape=}")
EVI_TS_df.head(2)

# %%
counter = 0
indeks = "NDVI"
for a_poly in IDs:
    if (counter % 500 == 0):
        print (f"{counter=}")

    curr_field = NDVI_TS_df[NDVI_TS_df[IDcolName]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[indeks].values, window_length=7, polyorder=3)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    if counter == 0:
        print(curr_field.head(2))
        print ("==================================================================================")
        print(f"{curr_field.index[0:5]=}")
        print ("==================================================================================")
        # print (NDVI_TS_df.loc[curr_field.index, ])
        # print ("==================================================================================")
        print(f"{len(SG)=}")
        print ("==================================================================================")
        print (SG[1:10])

    NDVI_TS_df.loc[curr_field.index, indeks] = SG
    counter += 1

NDVI_TS_df.drop_duplicates(inplace=True)
NDVI_TS_df.dropna(inplace=True)

del(indeks, curr_field, a_poly)
print ()
print ("=============== done ===============")
print (f"{NDVI_TS_df.shape=}")
NDVI_TS_df.head(2)

# %%
out_name = VI_TS_dir + "EVI_SG_WSUStudentSurvey2022.csv"
EVI_TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
EVI_TS_df.to_csv(out_name, index = False)

out_name = VI_TS_dir + "NDVI_SG_WSUStudentSurvey2022.csv"
NDVI_TS_df.sort_values(by=["ID", "human_system_start_time"], inplace=True)
NDVI_TS_df.to_csv(out_name, index = False)

# %%

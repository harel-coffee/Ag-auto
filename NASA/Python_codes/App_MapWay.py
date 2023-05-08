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

# %% [markdown]
# Rather than applying each step to all fields. let us do all steps to a given field, end-to-end, and then use ```map()```. 
#
# We can also use ```map()``` at each step on a given field. This way if one step is complete, then we have the output of that step and we do not have to start from the beginning!

# %%
b = "damn"
def myfunc(a):
    return a + b

x = map(myfunc, ('apple', 'banana', 'cherry'))
print(list(x))

# %%
import pandas as pd
import numpy as np
import os, sys

import time, datetime
from datetime import date, datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core_inPlace as nc

# %%
VI_TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
SF_data_dir = "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"

# %%
VI_idx = "EVI"

# %%
file_names = ["L7_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv", 
              "L8_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv",
              "L7_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv", 
              "L8_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv",
              "L7_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv",
              "L8_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv"]

VI_TS = pd.DataFrame()
for a_file in file_names:
    df = pd.read_csv(VI_TS_dir+a_file)
    df.dropna(subset=["EVI"], inplace=True)
    df = nc.add_human_start_time_by_system_start_time(df)
    
    if ("Adam" in a_file) or ("Benton" in a_file):
        df=df[df.human_system_start_time.dt.year==2016]
    
    if ("Grant" in a_file):
        df=df[df.human_system_start_time.dt.year==2017]

    if ("Franklin" in a_file) or ("Yakima" in a_file):
        df=df[df.human_system_start_time.dt.year==2018]

    VI_TS = pd.concat([VI_TS, df])
    
del(df)
len(VI_TS.ID.unique())

# %%
file_names = os.listdir(SF_data_dir)
file_names = [x for x in file_names if x.endswith(".csv") ]

SF_data = pd.DataFrame()
for a_file in file_names:
    df = pd.read_csv(SF_data_dir+a_file)
    SF_data = pd.concat([SF_data, df])

large_fields = SF_data[SF_data.ExctAcr>10]
large_fields.reset_index(drop=True, inplace=True)
large_fields.head(2)
print (f"{len(SF_data.ID.unique())=}")

del(df)

# %%
VI_TS = VI_TS[VI_TS.ID.isin(list(large_fields.ID.unique()))]
print (f"{len(VI_TS.ID.unique())=}")

# %%
VI_TS = VI_TS[["ID", VI_idx, "human_system_start_time"]]
VI_TS.sort_values(by=["ID", "human_system_start_time"], inplace=True)
VI_TS.reset_index(drop=True, inplace=True)

# %%
print (f"{VI_TS.shape=}")
VI_TS = nc.initial_clean(df=VI_TS, column_to_be_cleaned=VI_idx)
print (f"{VI_TS.shape=}")

# %%
IDs = VI_TS.ID.unique()
a_ID = IDs[3]

# %%
curr_field = VI_TS[VI_TS.ID==a_ID]
curr_field.head(2)


# %%
def interpolate_outliers_EVI_NDVI(outlier_input, f_ID, given_col):
    """
    outliers are those that are beyond boundaries. For example and EVI value of 2.
    Big jump in the other function means we have a big jump but we are still
    within the region of EVI values. If in 20 days we have a jump of 0.3 then that is noise.
    """

    # 1st block
    time_vec = outlier_input[outlier_input.ID==f_ID, "human_system_start_time"].values
    vec =      outlier_input[outlier_input.ID==f_ID, given_col].values

    # find out where are outliers
    high_outlier_inds = np.where(vec > 1)[0]
    low_outlier_inds = np.where(vec < -1)[0]

    all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
    all_outliers_idx = np.sort(all_outliers_idx)
    non_outiers = np.arange(len(vec))[~np.in1d(np.arange(len(vec)), all_outliers_idx)]

    # 2nd block
    if len(all_outliers_idx) == 0:
        return outlier_input

    """
    it is possible that for a field we only have x=2 data points
    where all the EVI/NDVI is outlier. Then, there is nothing to 
    use for interpolation. So, we return an empty datatable
    """
    if len(all_outliers_idx) == len(outlier_input):
        # outlier_input = initial clean(df=outlier_input, column_to_be_cleaned=given_col)
        outlier_input = outlier_input[outlier_input[given_col] < 1.5]
        outlier_input = outlier_input[outlier_input[given_col] > -1.5]
        return outlier_input

    # 3rd block
    # Get rid of outliers that are at the beginning of the time series
    # if len(non_outiers) > 0 :
    if non_outiers[0] > 0:
        vec[0 : non_outiers[0]] = vec[non_outiers[0]]

        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds  = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[~np.in1d(np.arange(len(vec)), all_outliers_idx)]
        if len(all_outliers_idx) == 0:
            outlier_input[given_col] = vec
            return outlier_input

    # 4th block
    # Get rid of outliers that are at the end of the time series
    if non_outiers[-1] < (len(vec) - 1):
        vec[non_outiers[-1] :] = vec[non_outiers[-1]]

        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[~np.in1d(np.arange(len(vec)), all_outliers_idx)]
        if len(all_outliers_idx) == 0:
            outlier_input[given_col] = vec
            return outlier_input
    """
    At this point outliers are in the middle of the vector
    and beginning and the end of the vector are clear.
    """
    for out_idx in all_outliers_idx:
        """
        Right here at the beginning we should check
        if vec[out_idx] is outlier or not. The reason is that
        there might be consecutive outliers at position m and m+1
        and we fix the one at m+1 when we are fixing m ...
        """
        # if ~(vec[out_idx] <= 1 and vec[out_idx] >= -1):
        if vec[out_idx] >= 1 or vec[out_idx] <= -1:
            left_pointer = out_idx - 1
            right_pointer = out_idx + 1
            while ~(vec[right_pointer] <= 1 and vec[right_pointer] >= -1):
                right_pointer += 1

            # form the line and fill in the outlier valies
            x1, y1 = time_vec[left_pointer], vec[left_pointer]
            x2, y2 = time_vec[right_pointer], vec[right_pointer]

            time_diff = x2 - x1
            y_diff = y2 - y1

            slope = y_diff / time_diff.astype(pd.Timedelta)
            intercept = y2 - (slope * int(x2))
            vec[left_pointer + 1 : right_pointer] = (
                slope * ((time_vec[left_pointer + 1 : right_pointer]).astype(int)) + intercept
            )
    outlier_input[given_col] = vec
    return outlier_input


# %%
# if curr_field.shape[0] > 2:

# %%

# %%
curr_field = VI_TS[VI_TS.ID==a_ID]
curr_field.head(2)


# %%

# %%
def expon(df, an_ID, VI_idx):
    df.loc[df.ID==a_ID, VI_idx]=df.loc[df.ID==a_ID, VI_idx]**2


# %%
expon(VI_TS, a_ID, VI_idx)

# %%

# %%

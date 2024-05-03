import numpy as np
import pandas as pd

import time, datetime
from pprint import pprint
import os, os.path, sys
import scipy
from scipy.linalg import inv

"""
There are scipy.linalg.block_diag() and scipy.sparse.block_diag()
"""


def GLS(X, y, weight_):
    """
    This function returns a generalized least square solution
    where the weight matrix is inverted in analytical form.
    """
    return (
        inv(X.T.values @ inv(weight_.values) @ X.values)
        @ X.T.values
        @ inv(weight_.values)
        @ y
    )


def WLS(X, y, weight_):
    """
    This function returns a weighted least square solution
    where the weight matrix is not inverted in analytical form.
    The weight matrix is already degined as inverse of the diagonal matrix
    where each random variable has different variance but they are un-correlated.
    """
    return inv(X.T.values @ weight_.values @ X.values) @ X.T.values @ weight_.values @ y


def create_adj_weight_matrix(data_df, adj_df, fips_var="state_fips"):
    """
    We need to make a block-diagonal weight matrix
    out of adjacency matrix.

    We need data_df and fips_var since the data might not be complete.
    For example, an state might be missing from a given year. So, we cannot
    use the adj_df in full.

    We need fips_var for distinguishing state or county

    data_df : dataframe of data to use for iterating over years.
    fips_var : string
    adj_df : dataframe of adjacency matrix
    """

    # Sort data
    data_df.sort_values(by=[fips_var, "year"], inplace=True)

    blocks = []
    diag_col_names = []
    for a_year in data_df.year.unique():
        curr_df = data_df[data_df.year == a_year]

        assert len(curr_df[fips_var].unique()) == len(curr_df[fips_var])
        curr_fips = list(curr_df[fips_var].unique())
        curr_adj_block = adj_df.loc[curr_fips, curr_fips].copy()
        assert (curr_adj_block.columns == curr_adj_block.index).all()

        blocks.append(curr_adj_block)
        diag_col_names = diag_col_names + list(curr_adj_block.columns)
    # the * allows to use a list.

    diag_block = scipy.linalg.block_diag(*blocks)
    diag_block = pd.DataFrame(diag_block)

    # rename columns so we know what's what
    diag_block.columns = diag_col_names
    diag_block.index = diag_col_names

    return diag_block


def convert_lb_2_kg(df, matt_total_npp_col, new_col_name):
    """
    Convert weight in lb to kg
    """
    df[new_col_name] = df[matt_total_npp_col] / 2.2046226218
    return df


def convert_lbperAcr_2_kg_in_sqM(df, matt_unit_npp_col, new_col_name):
    """
    Convert lb/acr to kg/m2

    1 acre is 4046.86 m2
    1 lb is 0.453592 kg (multiplying by 0.453592 is the same as diving by 2.205)
    Or just multiply by 0.000112085
    """
    # lb_2_kg = df[matt_unit_npp_col] / 2.205
    # lbAcr_2_kgm2 = lb_2_kg / 4046.86
    # df[new_col_name] = lbAcr_2_kgm2

    df[new_col_name] = df[matt_unit_npp_col] * 0.000112085
    return df


def add_lags_avg(df, lag_vars_, year_count, fips_name):
    """
    This function adds lagged variables in the sense of average.
    if year_count is 3, then average of past year, 2 years before, and 3 years before
          are averaged.
    df : pandas dataframe
    lag_vars_ : list of variable/column names to create the lags for
    year_count : integer: number of lag years we want.
    fips_name : str : name of column of fips; e.g. state_fips/county_fips
    """
    df_lag = df[["year", fips_name] + lag_vars_]
    df_lag = df_lag.groupby([fips_name]).rolling(year_count, on="year").mean()
    df_lag.reset_index(drop=False, inplace=True)
    df_lag.drop(columns=["level_1"], inplace=True)
    df_lag.dropna(subset=lag_vars_, inplace=True)

    df_lag["year"] = df_lag["year"] + 1
    df_lag.reset_index(inplace=True, drop=True)

    for a_col in lag_vars_:
        new_col = a_col + "_lagAvg" + str(year_count)
        df_lag.rename(columns={a_col: new_col}, inplace=True)
        df_lag.dropna(subset=[new_col], inplace=True)

    df = pd.merge(df, df_lag, on=["year", fips_name], how="left")
    df.dropna(subset=new_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_lags(df, merge_cols, lag_vars_, year_count):
    """
    This function adds lagged variables.
    df : pandas dataframe
    merge_cols : list of column names to merge on: state_fips/county_fips, year
    lag_vars_ : list of variable/column names to create the lags for
    year_count : integer: number of lag years we want.
    """
    cc_ = merge_cols + lag_vars_
    for yr_lag in np.arange(1, year_count + 1):
        df_needed_yrbefore = df[cc_].copy()
        df_needed_yrbefore["year"] = df_needed_yrbefore["year"] + yr_lag
        lag_col_names = [x + "_lag" + str(yr_lag) for x in lag_vars_]
        df_needed_yrbefore.columns = merge_cols + lag_col_names

        df = pd.merge(df, df_needed_yrbefore, on=merge_cols, how="left")
    return df


def compute_herbRatio_totalArea(hr):
    """
    We want to use average herb ratio and pixel count
    to compute total herb space.
    """
    pixel_length = 250
    pixel_area = pixel_length**2
    hr["herb_area_m2"] = pixel_area * hr["pixel_count"] * (hr["herb_avg"] / 100)

    # convert to acres for sake of consistency
    hr["herb_area_acr"] = hr["herb_area_m2"] / 4047
    hr.drop(labels=["herb_area_m2"], axis=1, inplace=True)
    return hr


def covert_totalNpp_2_unit(NPP_df, npp_total_col_, area_m2_col_, npp_unit_col_name_):
    """
    Min has unit NPP on county level.

    So, for state level, we have to compute total NPP first
    and then unit NPP for the state.

    Convert the total NPP to unit NPP.
    Total area can be area of rangeland in a county or an state

    Units are Kg * C / m^2

    1 m^2 = 0.000247105 acres
    """
    NPP_df[npp_unit_col_name_] = NPP_df[npp_total_col_] / NPP_df[area_m2_col_]
    return NPP_df


def covert_MattunitNPP_2_total(NPP_df, npp_unit_col_, acr_area_col_, npp_total_col_):
    """
    Convert the unit NPP to total area.
    Total area can be area of rangeland in a county or an state

    Units are punds per acre

    Arguments
    ---------
    NPP_df : dataframe
           whose one column is unit NPP

    npp_unit_col_ : str
           name of the unit NPP column

    acr_area_col_ : str
           name of the column that gives area in acres

    npp_area_col_ : str
           name of new column that will have total NPP

    Returns
    -------
    NPP_df : dataframe
           the dataframe that has a new column in it: total NPP

    """
    NPP_df[npp_total_col_] = NPP_df[npp_unit_col_] * NPP_df[acr_area_col_]

    return NPP_df


def covert_unitNPP_2_total(NPP_df, npp_unit_col_, acr_area_col_, npp_area_col_):
    """
    Convert the unit NPP to total area.
    Total area can be area of rangeland in a county or an state

    Units are Kg * C / m^2

    1 m^2 = 0.000247105 acres

    Arguments
    ---------
    NPP_df : dataframe
           whose one column is unit NPP

    npp_unit_col_ : str
           name of the unit NPP column

    acr_area_col_ : str
           name of the column that gives area in acres

    npp_area_col_ : str
           name of new column that will have total NPP

    Returns
    -------
    NPP_df : dataframe
           the dataframe that has a new column in it: total NPP

    """
    meterSq_to_acr = 0.000247105
    acr_2_m2 = 4046.862669715303
    NPP_df["area_m2"] = NPP_df[acr_area_col_] * acr_2_m2
    NPP_df[npp_area_col_] = NPP_df[npp_unit_col_] * NPP_df["area_m2"]
    # NPP_df[npp_area_col_] = (
    #     NPP_df[npp_unit_col_] * NPP_df[acr_area_col_]
    # ) / meterSq_to_acr
    return NPP_df


def census_stateCntyAnsi_2_countyFips(
    df, state_fip_col="state_ansi", county_fip_col="county_ansi"
):
    df[state_fip_col] = df[state_fip_col].astype("int32")
    df[county_fip_col] = df[county_fip_col].astype("int32")

    df[state_fip_col] = df[state_fip_col].astype("str")
    df[county_fip_col] = df[county_fip_col].astype("str")

    for idx in df.index:
        if len(df.loc[idx, state_fip_col]) == 1:
            df.loc[idx, state_fip_col] = "0" + df.loc[idx, state_fip_col]

        if len(df.loc[idx, county_fip_col]) == 1:
            df.loc[idx, county_fip_col] = "00" + df.loc[idx, county_fip_col]
        elif len(df.loc[idx, county_fip_col]) == 2:
            df.loc[idx, county_fip_col] = "0" + df.loc[idx, county_fip_col]

    df["county_fips"] = df[state_fip_col] + df[county_fip_col]
    return df


def clean_census(df, col_, col_to_lower=True):
    """
    Census data is weird;
        - Column can have ' (D)' or ' (Z)' in it.
        - Numbers are as strings.
    """
    if col_to_lower == True:
        df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
        col_ = col_.lower()
    if "state" in df.columns:
        df.state = df.state.str.title()
    if "county" in df.columns:
        df.county = df.county.str.title()

    df.reset_index(drop=True, inplace=True)

    """
    It is possible that this column is all numbers. 
    So, I put the following If there. 
    I am not sure how many cases are possible!!!
    So, maybe I should convert it to str first!
    But, then we might have produced NaN and who knows how many different
    patterns!!!
    """
    if df[col_].dtype == "O" or df[col_].dtype == "str":
        # df = df[df[col_] != " (D)"]
        # df = df[df[col_] != " (Z)"]
        # df = df[~(df[col_].str.contains(pat="(D)", case=False))]
        # df = df[~(df[col_].str.contains(pat="(Z)", case=False))]

        df = df[~(df[col_].str.contains(pat="(D)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(Z)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(S)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(NA)", case=False, na=False))]

    df.reset_index(drop=True, inplace=True)

    # this is not good condition. maybe the first one is na whose
    # type would be float.

    # if type(df[col_][0]) == str:
    #     df[col_] = df[col_].str.replace(",", "")
    #     df[col_] = df[col_].astype(float)

    df[col_] = df[col_].astype(str)
    df[col_] = df[col_].str.replace(",", "")
    df[col_] = df[col_].astype(float)

    if (
        ("state_ansi" in df.columns)
        and ("county_ansi" in df.columns)
        and not ("county_fips" in df.columns)
    ):
        df = census_stateCntyAnsi_2_countyFips(df)

    return df


def correct_Mins_county_6digitFIPS(df, col_):
    """
    Min has added a leading 1 to FIPS
    since some FIPs starts with 0.

    Get rid of 1 and convert to strings.
    """
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1)

    ## if county name is missing, that is for
    ## all of state. or sth. drop them. They have ' ' in them, no NA!
    if "county_name" in df.columns:
        df = df[df.county_name != " "].copy()
        df.reset_index(drop=True, inplace=True)
    return df


def correct_2digit_countyStandAloneFips(df, col_):
    """
    If the leading digit is zero, it will be gone.
    So, stand alone county FIPS can end up being 2 digit.
    We add zero back and FIPS will be string.
    """
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if len(df.loc[idx, col_]) == 2:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
        if len(df.loc[idx, col_]) == 1:
            df.loc[idx, col_] = "00" + df.loc[idx, col_]
    return df


def correct_4digitFips(df, col_):
    """
    If the leading digit is zero, it will be gone.
    So, county FIPS can end up being 4 digit.
    We add zero back and FIPS will be string.
    """
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if len(df.loc[idx, col_]) == 4:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
    return df


def correct_3digitStateFips_Min(df, col_):
    # Min has an extra 1 in his data. just get rid of it.
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1, 3)
    return df


def correct_state_int_fips_to_str(df, col_):
    # Min has an extra 1 in his data. just get rid of it.
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if (len(df.loc[idx, col_])) == 1:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
    return df

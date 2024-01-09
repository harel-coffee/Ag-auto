import pandas as pd
import os, pickle
from datetime import datetime

#
#
#   Years:      1992-2017 (Gridmet data are from 1979)
#   Grid count: 146229 in 25 states of interest.
#   Seasons:    Season 1: Jan - Mar
#               Season 2: Apr - Jul
#               Season 3: Aug - Sep
#               Season 4: Oct - Dec
#
#
########################################################################
######
######      Directories
######
########################################################################
data_base = "/data/project/agaid/h.noorazar/rangeland/"
param_dir = data_base + "parameters/"

in_dir = data_base + "/seasonal_variables/01_mean_over_county/"
out_dir = data_base + "/seasonal_variables/02_merged_mean_over_county/"
os.makedirs(out_dir, exist_ok=True)
########################################################################
######
######      Body
######
########################################################################
all_data = pd.DataFrame()
all_files = [x for x in os.listdir(in_dir) if x.endswith(".csv")]

for a_file in all_files:
    df = pd.read_csv(in_dir + a_file)
    all_data = pd.concat([all_data, df])

all_data.to_csv(out_dir + "countyMean_seasonalVars.csv", index=False)

grids_25states = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
grids_25states = grids_25states[["state", "county", "county_fips"]]
grids_25states.drop_duplicates(inplace=True)
grids_25states.county_fips = grids_25states.county_fips.astype("str")

for idx in grids_25states.index:
    if len(grids_25states.loc[idx, "county_fips"]) == 4:
        grids_25states.loc[idx, "county_fips"] = "0" + grids_25states.loc[idx, "county_fips"]


all_data = pd.merge(all_data, grids_25states, on=["state", "county"], how="left")
all_data.to_csv(out_dir + "countyMean_seasonalVars_wFips.csv", index=False)


export_ = {
    "countyMean_seasonalVars_wFips": all_data,
    "source_code": "d_merge_countyAgg.py",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
filename = out_dir + "countyMean_seasonalVars_wFips.sav"
pickle.dump(export_, open(filename, "wb"))

###
### Reshape seasonal variables.
###
# 4 seasons will collapse into 1 row.
all_data = all_data[
    ["state", "county", "year", "season", "countyMean_avg_Tavg", "countyMean_total_precip"]
]

L = int(len(all_data) / 4)
cntyMean_seasonVars_wide = pd.DataFrame(columns=["state", "county", "year"], index=range(L))
cntyMean_seasonVars_wide.state = "A"
cntyMean_seasonVars_wide.county = "A"
cntyMean_seasonVars_wide.year = 0

print(f"{cntyMean_seasonVars_wide.shape = }")
cntyMean_seasonVars_wide.head(2)

season_list = ["S1", "S2", "S3", "S4"]
temp_list = ["countyMean_avg_Tavg"] * 4
temp_cols = [i + "_" + j for i, j in zip(season_list, temp_list)]

precip_list = ["countyMean_total_precip"] * 4
precip_cols = [i + "_" + j for i, j in zip(season_list, precip_list)]

cntyMean_seasonVars_wide[precip_cols + temp_cols] = -60
all_data["state_county_year"] = (
    all_data.state + "_" + all_data.county + "_" + all_data.year.astype("str")
)

all_data[(all_data.state == "Alabama") & (all_data.county == "Madison") & (all_data.year == 1979)]

#######
####### Make it wide
#######
wide_row_idx = 0
print(len(all_data.state_county_year.unique()))
for a_slice_patt in all_data.state_county_year.unique():
    a_slice = all_data[all_data.state_county_year == a_slice_patt]
    cntyMean_seasonVars_wide.loc[wide_row_idx, "year"] = a_slice.year.iloc[0]
    cntyMean_seasonVars_wide.loc[wide_row_idx, "state"] = a_slice.state.iloc[0]
    cntyMean_seasonVars_wide.loc[wide_row_idx, "county"] = a_slice.county.iloc[0]
    cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[0]] = a_slice[
        "countyMean_avg_Tavg"
    ].values[0]
    cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[1]] = a_slice[
        "countyMean_avg_Tavg"
    ].values[1]
    cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[2]] = a_slice[
        "countyMean_avg_Tavg"
    ].values[2]
    cntyMean_seasonVars_wide.loc[wide_row_idx, temp_cols[3]] = a_slice[
        "countyMean_avg_Tavg"
    ].values[3]
    cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[0]] = a_slice[
        "countyMean_total_precip"
    ].values[0]
    cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[1]] = a_slice[
        "countyMean_total_precip"
    ].values[1]
    cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[2]] = a_slice[
        "countyMean_total_precip"
    ].values[2]
    cntyMean_seasonVars_wide.loc[wide_row_idx, precip_cols[3]] = a_slice[
        "countyMean_total_precip"
    ].values[3]
    wide_row_idx += 1
    if wide_row_idx % 2000 == 0:
        print(wide_row_idx)

###########
########### Check
###########
cntyMean_seasonVars_wide = pd.merge(
    cntyMean_seasonVars_wide, grids_25states, on=["state", "county"], how="left"
)
cntyMean_seasonVars_wide.to_csv(out_dir + "countyMean_seasonalVars_wFips.csv", index=False)


filename = out_dir + "wide_seasonal_vars_cntyMean_wFips.sav"

export_ = {
    "wide_seasonal_vars_cntyMean_wFips": cntyMean_seasonVars_wide,
    "source_code": "d_merge_countyAgg",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))
## It took [25 minutes] to run this cell

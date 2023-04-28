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

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"
data_dir = dir_base + "/8th_intersected_2008_2018_EastIrr/03_jumps_removed/"
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"

# %%
indeks = "NDVI"
batch = str(22)
regular_window_size = 10
IDcolName = "ID"
print(f"{indeks=}")
print(f"{batch=}")


# %%
f_name = "NoJump_intersect_" + indeks + "_batchNumber" + str(batch) + "_JFD.csv"
# out_name = output_dir + indeks + "_regular_intersect_batchNumber" + batch + "_JFD.csv"

an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)

if "system_start_time" in an_EE_TS.columns:
    an_EE_TS.drop(["system_start_time"], axis=1, inplace=True)

an_EE_TS["human_system_start_time"] = pd.to_datetime(an_EE_TS["human_system_start_time"])
an_EE_TS["ID"] = an_EE_TS["ID"].astype(str)
print(an_EE_TS.head(2))


# %%
### List of unique polygons
###
ID_list = an_EE_TS[IDcolName].unique()
print(len(ID_list))

# %%
reg_cols = ["ID", "human_system_start_time", indeks]  # list(an_EE_TS.columns)
print(f"{reg_cols=}")

# %%
st_yr = an_EE_TS.human_system_start_time.dt.year.min()
end_yr = an_EE_TS.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366  # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size))  # no_days // regular_window_size
nrows = no_steps * len(ID_list)

output_df = pd.DataFrame(data=None, index=np.arange(nrows), columns=reg_cols)
print("st_yr is {}!".format(st_yr))
print("end_yr is {}!".format(end_yr))
print("nrows is {}!".format(nrows))

# %%
counter = 0
row_pointer = 0

# %%
a_poly = ID_list[0]

# %%
if counter % 1000 == 0:
    print(counter)
curr_field = an_EE_TS[an_EE_TS[IDcolName] == a_poly].copy()

# %%
# Sort by DoY (sanitary check)
curr_field.sort_values(by=["human_system_start_time"], inplace=True)
curr_field.reset_index(drop=True, inplace=True)

# %%
regularized_TS = nc.regularize_a_field(
        a_df=curr_field,
        V_idks=indeks,
        interval_size=regular_window_size,
        start_year=st_yr,
        end_year=end_yr,
    )

# %%
regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS=regularized_TS, V_idx=indeks)

# %%
if counter == 0:
    print(
        f"{list(output_df.columns) = }",
    )
    print(
        f"{list(regularized_TS.columns) = }",
    )

# %%

# %%

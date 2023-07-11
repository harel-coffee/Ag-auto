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
import os, sys, pickle

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/8th_intersected_2008_2018_EastIrr/00_raw/"

# %% [markdown]
# ### Check problems...
# but we have to make sure if these are the problems when we started doing batches of size 500, since the name
# is not indicative of anything!!

# %%
problems_L5_early = pickle.load(open(data_dir_base + "problems_L5_early.pkl", "rb"))
problems_L7 = pickle.load(open(data_dir_base + "problems_L7.pkl", "rb"))

# %%

# %%
csv_files = [x for x in os.listdir(data_dir_base + "intersection_pre2008/") if x.endswith(".csv")]
csv_L4 = [x for x in os.listdir(data_dir_base + "intersection_pre2008_L4/") if x.endswith(".csv")]
csv_L5_early = [x for x in os.listdir(data_dir_base + "intersection_pre2008_L5_early/") if x.endswith(".csv")]
csv_L5_late = [x for x in os.listdir(data_dir_base + "intersection_pre2008_L5_late/") if x.endswith(".csv")]
csv_L7 = [x for x in os.listdir(data_dir_base + "intersection_pre2008_L7/") if x.endswith(".csv")]

# %%
csv_L4 = [x for x in csv_L4 if ("BS500" in x)]
csv_L5_early = [x for x in csv_L5_early if ("BS500" in x)]
csv_L5_late = [x for x in csv_L5_late if ("BS500" in x)]
csv_L7 = [x for x in csv_L7 if ("BS500" in x)]

# %%
print(f"{len(csv_L4) = }")
print(f"{len(csv_L5_early) = }")
print(f"{len(csv_L5_late) = }")
print(f"{len(csv_L7) = }")

# %%
# Check if problems are real
print (problems_L5_early)
print (problems_L7)

# %%
print ("L5_early_1991-01-01_1992-01-01_BS500_block79.csv" in csv_L5_early)
print ("L7_2007-01-01_2008-01-01_BS500_block15.csv" in csv_L7)

# %%
# %%time
### Why "L4_1993-01-01_1994-01-01_BS500_block18.csv" is empty?

# pre-allocate. 500 fields per file. say 100 data per field.

L4 = pd.DataFrame(columns=["ID", "EVI", "NDVI", "system_start_time"], index=range(len(csv_L4)*500*100))
L5_early = pd.DataFrame(columns=["ID", "EVI", "NDVI", "system_start_time"], index=range(len(csv_L5_early)*500*100))
L5_late = pd.DataFrame(columns=["ID", "EVI", "NDVI", "system_start_time"], index=range(len(csv_L5_late)*500*100))
L7 = pd.DataFrame(columns=["ID", "EVI", "NDVI", "system_start_time"], index=range(len(csv_L7)*500*100))

# %%
# %%time

L4_empty_counter, pointer=0, 0
for a_file in csv_L4:
    try:
        curr_df = pd.read_csv( data_dir_base + "intersection_pre2008_L4/" + a_file)
        L4[pointer:pointer+len(curr_df)] = curr_df
        pointer = pointer + len(curr_df)
    except:
        L4_empty_counter+=1

L5_early_empty_counter, pointer=0, 0
for a_file in csv_L5_early:
    try:
        curr_df = pd.read_csv(data_dir_base + "intersection_pre2008_L5_early/" + a_file)
        L5_early[pointer:pointer+len(curr_df)] = curr_df
        pointer = pointer + len(curr_df)
    except:
        L5_early_empty_counter+=1

L5_late_empty_counter, pointer=0, 0
for a_file in csv_L5_late:
    try:
        curr_df = pd.read_csv(data_dir_base + "intersection_pre2008_L5_late/" + a_file)
        L5_late[pointer:pointer+len(curr_df)] = curr_df
        pointer = pointer + len(curr_df)

    except:
        L5_late_empty_counter+=1

L7_empty_counter, pointer=0, 0
for a_file in csv_L7:
    try:
        curr_df = pd.read_csv(data_dir_base + "intersection_pre2008_L7/" + a_file)
        L7[pointer:pointer+len(curr_df)] = curr_df
        pointer = pointer + len(curr_df)
    except:
        L7_empty_counter+=1
        
print (f"{L4_empty_counter = }")
print (f"{L5_early_empty_counter = }")
print (f"{L5_late_empty_counter = }")
print (f"{L7_empty_counter = }")
print ()

# %%

# %%
# %%time
L4.dropna(axis='index', inplace=True, how="all")
L5_early.dropna(axis='index', inplace=True, how="all")
L5_late.dropna(axis='index', inplace=True, how="all")
L7.dropna(axis='index', inplace=True, how="all")

# %%
print (f"{L4.shape = }")
print (f"{L5_early.shape = }")
print (f"{L5_late.shape = }")
print (f"{L7.shape = }")
print ()

# %%
# %%time
out_name = data_dir_base + "L4_pre2008.csv"
L4.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_early_pre2008.csv"
L5_early.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_late_pre2008.csv"
L5_late.to_csv(out_name, index = False)

out_name = data_dir_base + "L7_pre2008.csv"
L7.to_csv(out_name, index = False)

# %%
L4_EVI = L4.copy()
L4_EVI = L4_EVI.drop(columns=["NDVI"])
L4_EVI.dropna(subset=["EVI"], inplace=True)

L4_NDVI = L4.copy()
L4_NDVI = L4_NDVI.drop(columns=["EVI"])
L4_NDVI.dropna(subset=["NDVI"], inplace=True)

L4_NDVI.sort_values(by=["ID", "system_start_time"], inplace=True)
L4_EVI.sort_values(by =["ID", "system_start_time"], inplace=True)

L4_EVI.reset_index(drop=True, inplace=True)
L4_NDVI.reset_index(drop=True, inplace=True)

# %%
L7_EVI = L7.copy()
L7_EVI = L7_EVI.drop(columns=["NDVI"])
L7_EVI.dropna(subset=["EVI"], inplace=True)

L7_NDVI = L7.copy()
L7_NDVI = L7_NDVI.drop(columns=["EVI"])
L7_NDVI.dropna(subset=["NDVI"], inplace=True)

L7_NDVI.sort_values(by=["ID", "system_start_time"], inplace=True)
L7_EVI.sort_values(by=["ID", "system_start_time"], inplace=True)

L7_EVI.reset_index(drop=True, inplace=True)
L7_NDVI.reset_index(drop=True, inplace=True)

# %%

# %%
L5_early_EVI = L5_early.copy()
L5_early_EVI = L5_early_EVI.drop(columns=["NDVI"])
L5_early_EVI.dropna(subset=["EVI"], inplace=True)

L5_early_NDVI = L5_early.copy()
L5_early_NDVI = L5_early_NDVI.drop(columns=["EVI"])
L5_early_NDVI.dropna(subset=["NDVI"], inplace=True)

L5_early_NDVI.sort_values(by=["ID", "system_start_time"], inplace=True)
L5_early_EVI.sort_values(by=["ID", "system_start_time"], inplace=True)

L5_early_EVI.reset_index(drop=True, inplace=True)
L5_early_NDVI.reset_index(drop=True, inplace=True)

# %%
L5_late_EVI = L5_late.copy()
L5_late_EVI = L5_late_EVI.drop(columns=["NDVI"])
L5_late_EVI.dropna(subset=["EVI"], inplace=True)

L5_late_NDVI = L5_late.copy()
L5_late_NDVI = L5_late_NDVI.drop(columns=["EVI"])
L5_late_NDVI.dropna(subset=["NDVI"], inplace=True)

L5_late_NDVI.sort_values(by=["ID", "system_start_time"], inplace=True)
L5_late_EVI.sort_values(by=["ID", "system_start_time"], inplace=True)

L5_late_EVI.reset_index(drop=True, inplace=True)
L5_late_NDVI.reset_index(drop=True, inplace=True)

# %% [markdown]
# ## Add human time

# %%
nc.add_human_start_time_by_system_start_time(L4_EVI)
nc.add_human_start_time_by_system_start_time(L4_NDVI)

# %%
nc.add_human_start_time_by_system_start_time(L5_early_EVI)
nc.add_human_start_time_by_system_start_time(L5_early_NDVI)

nc.add_human_start_time_by_system_start_time(L5_late_EVI)
nc.add_human_start_time_by_system_start_time(L5_late_NDVI)

nc.add_human_start_time_by_system_start_time(L7_EVI)
nc.add_human_start_time_by_system_start_time(L7_NDVI)

# %%
# %%time

out_name = data_dir_base + "L4_EVI_pre2008.csv"
L4_EVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L4_NDVI_pre2008.csv"
L4_NDVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_early_EVI_pre2008.csv"
L5_early_EVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_early_NDVI_pre2008.csv"
L5_early_NDVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_late_EVI_pre2008.csv"
L5_late_EVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L5_late_NDVI_pre2008.csv"
L5_late_NDVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L7_EVI_pre2008.csv"
L7_EVI.to_csv(out_name, index = False)

out_name = data_dir_base + "L7_NDVI_pre2008.csv"
L7_NDVI.to_csv(out_name, index = False)

# %%
# %%time

out_name = data_dir_base + "L4_EVI_pre2008.csv"
L4_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L4_NDVI_pre2008.csv"
L4_NDVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_early_EVI_pre2008.csv"
L5_early_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_early_NDVI_pre2008.csv"
L5_early_NDVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_late_EVI_pre2008.csv"
L5_late_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_late_NDVI_pre2008.csv"
L5_late_NDVI = pd.read_csv(out_name)

out_name = data_dir_base + "L7_EVI_pre2008.csv"
L7_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L7_NDVI_pre2008.csv"
L7_NDVI = pd.read_csv(out_name)

# %%
L457_EVI_pre2008 = pd.concat([L4_EVI, L5_early_EVI, L5_late_EVI, L7_EVI])
L457_NDVI_pre2008 = pd.concat([L4_NDVI, L5_early_NDVI, L5_late_NDVI, L7_NDVI])

# %%
L457_EVI_pre2008["human_system_start_time"] = pd.to_datetime(L457_EVI_pre2008["human_system_start_time"])
L457_NDVI_pre2008["human_system_start_time"] = pd.to_datetime(L457_NDVI_pre2008["human_system_start_time"])

# %%
L457_EVI_pre2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)
L457_NDVI_pre2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)

# %%
L457_EVI_pre2008.reset_index(drop=True, inplace=True)
L457_NDVI_pre2008.reset_index(drop=True, inplace=True)

# %%
out_name = data_dir_base + "L457_EVI_pre2008.csv"
L457_EVI_pre2008.to_csv(out_name, index = False)

out_name = data_dir_base + "L457_NDVI_pre2008.csv"
L457_NDVI_pre2008.to_csv(out_name, index = False)

# %%
L457_EVI_pre2008.head(2)

# %%

# %%

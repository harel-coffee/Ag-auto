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
import seaborn as sns
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/8th_intersected_2008_2018_EastIrr/00_raw/"

# %%
# # %%time
# out_name = data_dir_base + "L4_pre2008.csv"
# L4 = pd_read_csv(out_name)

# out_name = data_dir_base + "L5_early_pre2008.csv"
# L5_early = pd_read_csv(out_name)

# out_name = data_dir_base + "L5_late_pre2008.csv"
# L5_late = pd_read_csv(out_name)

# out_name = data_dir_base + "L7_pre2008.csv"
# L7 = pd_read_csv(out_name)

# %%
out_name = data_dir_base + "L4_EVI_pre2008.csv"
L4_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L4_NDVI_pre2008.csv"
L4_NDVI = pd.read_csv(out_name)

# %%
out_name = data_dir_base + "L5_early_EVI_pre2008.csv"
L5_early_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_early_NDVI_pre2008.csv"
L5_early_NDVI = pd.read_csv(out_name)

# %%
out_name = data_dir_base + "L5_late_EVI_pre2008.csv"
L5_late_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L5_late_NDVI_pre2008.csv"
L5_late_NDVI = pd.read_csv(out_name)

# %%
out_name = data_dir_base + "L7_EVI_pre2008.csv"
L7_EVI = pd.read_csv(out_name)

out_name = data_dir_base + "L7_NDVI_pre2008.csv"
L7_NDVI = pd.read_csv(out_name)

# %% [markdown]
# ## Add year

# %%
L7_NDVI["human_system_start_time"] = pd.to_datetime(L7_NDVI["human_system_start_time"])
L7_EVI["human_system_start_time"] = pd.to_datetime(L7_EVI["human_system_start_time"])

L4_EVI["human_system_start_time"] = pd.to_datetime(L4_EVI["human_system_start_time"])
L4_NDVI["human_system_start_time"] = pd.to_datetime(L4_NDVI["human_system_start_time"])

L5_early_NDVI["human_system_start_time"] = pd.to_datetime(L5_early_NDVI["human_system_start_time"])
L5_early_EVI["human_system_start_time"] = pd.to_datetime(L5_early_EVI["human_system_start_time"])

L5_late_NDVI["human_system_start_time"] = pd.to_datetime(L5_late_NDVI["human_system_start_time"])
L5_late_EVI["human_system_start_time"] = pd.to_datetime(L5_late_EVI["human_system_start_time"])

# %%
L4_EVI["year"]  = L4_EVI['human_system_start_time'].dt.year
L4_NDVI["year"] = L4_NDVI['human_system_start_time'].dt.year

L5_early_EVI["year"] = L5_early_EVI['human_system_start_time'].dt.year
L5_early_NDVI["year"] = L5_early_NDVI['human_system_start_time'].dt.year

L5_late_EVI["year"] = L5_late_EVI['human_system_start_time'].dt.year
L5_late_NDVI["year"] = L5_late_NDVI['human_system_start_time'].dt.year

L7_EVI["year"] = L7_EVI['human_system_start_time'].dt.year
L7_NDVI["year"] = L7_NDVI['human_system_start_time'].dt.year

# %%
L7_NDVI.head(2)

# %% [markdown]
# #### See the stat per year fields

# %%
print (f'{L4_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print (f'{L4_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print (f'{L4_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print (f'{L4_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')
print("----------------------------------------------------------------------")

print (f'{L5_early_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print (f'{L5_early_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print ()
print (f'{L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print (f'{L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')


print("----------------------------------------------------------------------")
print (f'{L5_late_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print (f'{L5_late_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print ()
print (f'{L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print (f'{L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')

print("----------------------------------------------------------------------")
print (f'{L7_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print (f'{L7_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print (f'{L7_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print (f'{L7_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')

# %% [markdown]
# # We need to merge them together so that in each year 2 satellies are present.
# Then do stats. Merge them according to overlapping years.

# %%

# %%
# sns.countplot(x="NDVI",data=L4_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
sns.countplot(ax=axes[0], x="EVI", data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())
sns.countplot(ax=axes[1], x="NDVI", data=L4_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index());
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
# # L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index().value_counts().plot(ax=ax, kind='bar')
# L4_EVI_count = L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()

# fig, ax = plt.subplots()
# L4_EVI_count['EVI'].value_counts().plot(ax=ax, kind='bar')

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(ax=axes[0], x="EVI",  data= L7_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())
sns.countplot(ax=axes[1], x="NDVI", data=L7_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index());
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True);
sns.countplot(ax=axes[0], x="EVI",  data= L5_late_EVI.groupby(["ID", "year"])["EVI"].count().reset_index());
sns.countplot(ax=axes[1], x="NDVI", data=L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index());
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True);
sns.countplot(ax=axes[0], x="EVI",  data=L5_early_EVI.groupby(["ID", "year"])["EVI"].count().reset_index());
sns.countplot(ax=axes[1], x="NDVI", data=L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index());
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%

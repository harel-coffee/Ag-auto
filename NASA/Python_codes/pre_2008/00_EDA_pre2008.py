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
import os, sys
import seaborn as sns

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
data_dir_base = (
    "/Users/hn/Documents/01_research_data/NASA/VI_TS/8th_intersected_2008_2018_EastIrr/00_raw/"
)

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
L4_EVI["year"] = L4_EVI["human_system_start_time"].dt.year
L4_NDVI["year"] = L4_NDVI["human_system_start_time"].dt.year

L5_early_EVI["year"] = L5_early_EVI["human_system_start_time"].dt.year
L5_early_NDVI["year"] = L5_early_NDVI["human_system_start_time"].dt.year

L5_late_EVI["year"] = L5_late_EVI["human_system_start_time"].dt.year
L5_late_NDVI["year"] = L5_late_NDVI["human_system_start_time"].dt.year

L7_EVI["year"] = L7_EVI["human_system_start_time"].dt.year
L7_NDVI["year"] = L7_NDVI["human_system_start_time"].dt.year

# %%
L7_NDVI.head(2)

# %% [markdown]
# #### See the stat per year fields

# %%
print(f'{L4_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print(f'{L4_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print(f'{L4_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print(f'{L4_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')
print("----------------------------------------------------------------------")

print(f'{L5_early_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print(f'{L5_early_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print(f'{L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print(f'{L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')


print("----------------------------------------------------------------------")
print(f'{L5_late_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print(f'{L5_late_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print(f'{L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print(f'{L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')

print("----------------------------------------------------------------------")
print(f'{L7_EVI.groupby(["ID", "year"])["EVI"].count().min() = }')
print(f'{L7_EVI.groupby(["ID", "year"])["EVI"].count().max() = }')
print()
print(f'{L7_NDVI.groupby(["ID", "year"])["NDVI"].count().min() = }')
print(f'{L7_NDVI.groupby(["ID", "year"])["NDVI"].count().max() = }')

# %%

# %%
# sns.countplot(x="NDVI",data=L4_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
sns.countplot(ax=axes[0], x="EVI", data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())
sns.countplot(
    ax=axes[1], x="NDVI", data=L4_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
# # L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index().value_counts().plot(ax=ax, kind='bar')
# L4_EVI_count = L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()

# fig, ax = plt.subplots()
# L4_EVI_count['EVI'].value_counts().plot(ax=ax, kind='bar')

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(ax=axes[0], x="EVI", data=L7_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())
sns.countplot(
    ax=axes[1], x="NDVI", data=L7_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L5_late_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="NDVI", data=L5_late_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L5_early_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="NDVI", data=L5_early_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %% [markdown]
# # We need to merge them together so that in each year 2 satellies are present.
# Then do stats. Merge them according to overlapping years.

# %%
satellit_info = {
    "L4": ["LANDSAT/LT04/C02/T1_L2", 1984, 1993],
    "L5_early": ["LANDSAT/LT05/C02/T1_L2", 1984, 1993],
    "L5_late": ["LANDSAT/LT05/C02/T1_L2", 1999, 2007],
    "L7": ["LANDSAT/LE07/C02/T1_L2", 1999, 2007],
}

# %%
L45_EVI = pd.concat([L4_EVI, L5_early_EVI])
L57_EVI = pd.concat([L5_late_EVI, L7_EVI])

# %%
L45_EVI.sort_values(by=["ID", "human_system_start_time"], inplace=True)
L57_EVI.sort_values(by=["ID", "human_system_start_time"], inplace=True)

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L45_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="EVI", data=L57_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
L45_NDVI = pd.concat([L4_NDVI, L5_early_NDVI])
L57_NDVI = pd.concat([L5_late_NDVI, L7_NDVI])

L45_NDVI.sort_values(by=["ID", "human_system_start_time"], inplace=True)
L57_NDVI.sort_values(by=["ID", "human_system_start_time"], inplace=True)

# %%
fig, axes = plt.subplots(1, 2, figsize=(25, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="NDVI", data=L45_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="NDVI", data=L57_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %% [markdown]
# # Compare to post-2008

# %%
L5_post_2008 = pd.read_csv(
    data_dir_base + "L5_T1C2L2_inters_2008_2018_EastIrr_2008-01-01_2022-01-01.csv"
)
L7_post_2008 = pd.read_csv(
    data_dir_base + "L7_T1C2L2_inters_2008_2018_EastIrr_2008-01-01_2022-01-01.csv"
)
L8_post_2008 = pd.read_csv(
    data_dir_base + "L8_T1C2L2_inters_2008_2018_EastIrr_2008-01-01_2022-01-01.csv"
)

# %%
print(f"{L5_post_2008.shape=}")
print(f"{L5_post_2008.shape=}")

# %%
L5_post_2008_EVI = L5_post_2008[["ID", "EVI", "system_start_time"]].copy()
L5_post_2008_NDVI = L5_post_2008[["ID", "NDVI", "system_start_time"]].copy()

L7_post_2008_EVI = L7_post_2008[["ID", "EVI", "system_start_time"]].copy()
L7_post_2008_NDVI = L7_post_2008[["ID", "NDVI", "system_start_time"]].copy()

L8_post_2008_EVI = L8_post_2008[["ID", "EVI", "system_start_time"]].copy()
L8_post_2008_NDVI = L8_post_2008[["ID", "NDVI", "system_start_time"]].copy()

# %%
del (L5_post_2008, L7_post_2008, L8_post_2008)

# %%
L5_post_2008_EVI.dropna(subset=["EVI"], inplace=True)
L5_post_2008_NDVI.dropna(subset=["NDVI"], inplace=True)

L7_post_2008_EVI.dropna(subset=["EVI"], inplace=True)
L7_post_2008_NDVI.dropna(subset=["NDVI"], inplace=True)

L8_post_2008_EVI.dropna(subset=["EVI"], inplace=True)
L8_post_2008_NDVI.dropna(subset=["NDVI"], inplace=True)

# %%
nc.add_human_start_time_by_system_start_time(L5_post_2008_EVI)
nc.add_human_start_time_by_system_start_time(L5_post_2008_NDVI)

nc.add_human_start_time_by_system_start_time(L7_post_2008_EVI)
nc.add_human_start_time_by_system_start_time(L7_post_2008_NDVI)

nc.add_human_start_time_by_system_start_time(L8_post_2008_EVI)
nc.add_human_start_time_by_system_start_time(L8_post_2008_NDVI)

# %%
L5_post_2008_EVI["year"] = L5_post_2008_EVI["human_system_start_time"].dt.year
L5_post_2008_NDVI["year"] = L5_post_2008_NDVI["human_system_start_time"].dt.year

L7_post_2008_EVI["year"] = L7_post_2008_EVI["human_system_start_time"].dt.year
L7_post_2008_NDVI["year"] = L7_post_2008_NDVI["human_system_start_time"].dt.year

L8_post_2008_EVI["year"] = L8_post_2008_EVI["human_system_start_time"].dt.year
L8_post_2008_NDVI["year"] = L8_post_2008_NDVI["human_system_start_time"].dt.year

# %%
L5_post_2008_NDVI.reset_index(drop=True, inplace=True)
L5_post_2008_EVI.reset_index(drop=True, inplace=True)

L7_post_2008_NDVI.reset_index(drop=True, inplace=True)
L7_post_2008_EVI.reset_index(drop=True, inplace=True)

L8_post_2008_NDVI.reset_index(drop=True, inplace=True)
L8_post_2008_EVI.reset_index(drop=True, inplace=True)

# %%
L57_EVI_post_2008 = pd.concat([L5_post_2008_EVI, L7_post_2008_EVI[L7_post_2008_EVI.year <= 2011]])
L78_EVI_post_2008 = pd.concat([L8_post_2008_EVI, L7_post_2008_EVI[L7_post_2008_EVI.year > 2011]])

L57_EVI_post_2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)
L78_EVI_post_2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)

L57_EVI_post_2008.reset_index(drop=True, inplace=True)
L78_EVI_post_2008.reset_index(drop=True, inplace=True)

# %%
L57_NDVI_post_2008 = pd.concat(
    [L5_post_2008_NDVI, L7_post_2008_NDVI[L7_post_2008_NDVI.year <= 2011]]
)
L78_NDVI_post_2008 = pd.concat(
    [L8_post_2008_NDVI, L7_post_2008_NDVI[L7_post_2008_NDVI.year > 2011]]
)

L57_NDVI_post_2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)
L78_NDVI_post_2008.sort_values(by=["ID", "human_system_start_time"], inplace=True)

L57_NDVI_post_2008.reset_index(drop=True, inplace=True)
L78_NDVI_post_2008.reset_index(drop=True, inplace=True)

# %%
"""
     Post 2008
"""
fig, axes = plt.subplots(1, 2, figsize=(35, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L57_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="EVI", data=L78_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
print(len(L57_EVI_post_2008.ID.unique()) * len(L57_EVI_post_2008.year.unique()))
print(len(L78_EVI_post_2008.ID.unique()) * len(L78_EVI_post_2008.year.unique()))

# %%
"""
     Pre 2008
"""
fig, axes = plt.subplots(1, 2, figsize=(35, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L45_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="EVI", data=L57_EVI.groupby(["ID", "year"])["EVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
print(len(L45_EVI.ID.unique()) * len(L45_EVI.year.unique()))
print(len(L57_EVI.ID.unique()) * len(L57_EVI.year.unique()))

# %%
L57_EVI.year.min()

# %% [markdown]
# # Normalize by number of data points

# %%
"""
     Post 2008
"""
fig, axes = plt.subplots(1, 2, figsize=(35, 5), sharey=True)
sns.countplot(
    ax=axes[0], x="EVI", data=L57_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index()
)
sns.countplot(
    ax=axes[1], x="EVI", data=L78_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index()
)
# sns.countplot(x="",data=L4_EVI.groupby(["ID", "year"])["EVI"].count().reset_index())

# %%
L57_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index()

# %%
df = sns.load_dataset("titanic")
df.head()

# %%
df.groupby("class")["survived"].value_counts(normalize=True)

# %%
x, y = "class", "survived"

df.groupby("class")["survived"].value_counts(normalize=True).mul(100).rename(
    "percent"
).reset_index().pipe((sns.catplot, "data"), x=x, y="percent", hue=y, kind="bar")

# %%
L57_NDVI_post_2008_A = (
    L57_NDVI_post_2008.groupby(["ID", "year"])["NDVI"].count().rename("data_per_year").reset_index()
)

# A["data_per_year"].value_counts
L57_NDVI_post_2008_A = (
    L57_NDVI_post_2008_A.groupby(["data_per_year"])["data_per_year"]
    .count()
    .rename("count")
    .reset_index()
)
L57_NDVI_post_2008_A["percentage"] = 100 * (
    L57_NDVI_post_2008_A["count"] / L57_NDVI_post_2008_A["count"].sum()
)

# %%
L78_NDVI_post_2008_A = (
    L78_NDVI_post_2008.groupby(["ID", "year"])["NDVI"].count().rename("data_per_year").reset_index()
)

# A["data_per_year"].value_counts
L78_NDVI_post_2008_A = (
    L78_NDVI_post_2008_A.groupby(["data_per_year"])["data_per_year"]
    .count()
    .rename("count")
    .reset_index()
)
L78_NDVI_post_2008_A["percentage"] = 100 * (
    L78_NDVI_post_2008_A["count"] / L78_NDVI_post_2008_A["count"].sum()
)

# %%
L45_NDVI_A = L45_NDVI.groupby(["ID", "year"])["NDVI"].count().rename("data_per_year").reset_index()

# A["data_per_year"].value_counts
L45_NDVI_A = (
    L45_NDVI_A.groupby(["data_per_year"])["data_per_year"].count().rename("count").reset_index()
)
L45_NDVI_A["percentage"] = 100 * (L45_NDVI_A["count"] / L45_NDVI_A["count"].sum())

# %%
L57_NDVI_A = L57_NDVI.groupby(["ID", "year"])["NDVI"].count().rename("data_per_year").reset_index()

# A["data_per_year"].value_counts
L57_NDVI_A = (
    L57_NDVI_A.groupby(["data_per_year"])["data_per_year"].count().rename("count").reset_index()
)
L57_NDVI_A["percentage"] = 100 * (L57_NDVI_A["count"] / L57_NDVI_A["count"].sum())

# %%
tick_legend_FontSize = 20

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
"""
     Pre 2008
"""
fig, axes = plt.subplots(1, 2, figsize=(35, 5), sharey=True)
fig.tight_layout(pad=5.0)
axes[0].tick_params(labelrotation=90)
axes[1].tick_params(labelrotation=90)

sns.barplot(ax=axes[0], x="data_per_year", y="percentage", data=L45_NDVI_A)
sns.barplot(ax=axes[1], x="data_per_year", y="percentage", data=L57_NDVI_A)
plt.xticks(rotation=90)

# %%
"""
     post 2008
"""
fig, axes = plt.subplots(1, 2, figsize=(35, 5), sharey=True)
fig.tight_layout(pad=5.0)
# plt.xticks(rotation=90);
axes[0].tick_params(labelrotation=90)
axes[1].tick_params(labelrotation=90)

sns.barplot(ax=axes[0], x="data_per_year", y="percentage", data=L57_NDVI_post_2008_A)
sns.barplot(ax=axes[1], x="data_per_year", y="percentage", data=L78_NDVI_post_2008_A)

# %%

# %%
A = L45_NDVI.groupby(["ID", "year"])["NDVI"].count().rename("data_count").reset_index()
A[A.data_count <= 16].head(2)

# %%
A[A.data_count <= 16].groupby(["year", "data_count"]).count()

# %%
A = L57_NDVI_post_2008.groupby(["ID", "year"])["NDVI"].count().rename("data_count").reset_index()
A[A.data_count <= 16].head(2)

# %%
A[A.data_count <= 16].groupby(["year", "data_count"]).count()

# %%

# %%
# L57_EVI_post_2008.groupby(["ID", "year"])["EVI"].count().reset_index().value_counts(normalize=True)\
# .mul(100).rename('percent').reset_index()

# # A["count_percent"] = (A.counts / len(A))*100
# # A.head(10)

# %%
L4_EVI

# %%

# %%

# %%

# %%

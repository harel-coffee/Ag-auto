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


import sys, os, os.path

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc
import NASA_plot_core as ncp

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]

evaluation_set = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/parameters/evaluation_set.csv")

SF_train_unique_crops = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/SF_train_unique_crops.csv")

ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")
trainTest = pd.concat([train80, test20])
trainTest_1 = trainTest.copy()
trainTest_2 = trainTest.copy()

trainTest_1 = pd.merge(trainTest_1, meta, how="left", on="ID")
trainTest_2 = pd.merge(trainTest_2, evaluation_set, how="left", on="ID")
trainTest_2.equals(trainTest_1)
test20 = pd.merge(test20, meta, on=["ID"], how="left")
test20_apples = test20[test20.CropTyp == "apple"]

pred_dir = "/Users/hn/Documents/01_research_data/NASA/RegionalStatData/"

all_preds = pd.read_csv(pred_dir + "all_preds_overSample.csv")

all_preds.head(2)
test20_apples = pd.merge(test20_apples, all_preds[["ID", "DL_NDVI_SG_prob_point3"]], on=["ID"], how="left")
test20_apples[test20_apples.DL_NDVI_SG_prob_point3==2]


# %%

# %%
SF_data_dir = "/Users/hn/Documents/01_research_data/00_shapeFiles/01_shapefiles_data_part_not_filtered/"

# %%
WSDA_DataTable_2015 = pd.read_csv(SF_data_dir + "WSDA_DataTable_2015.csv")
WSDA_DataTable_2016 = pd.read_csv(SF_data_dir + "WSDA_DataTable_2016.csv")
WSDA_DataTable_2017 = pd.read_csv(SF_data_dir + "WSDA_DataTable_2017.csv")
WSDA_DataTable_2018 = pd.read_csv(SF_data_dir + "WSDA_DataTable_2018.csv")

# %%
WSDA_DataTable_2018.head(2)

# %%
WSDA_DataTable = pd.concat([WSDA_DataTable_2015[["ID", "ExctAcr", "Irrigtn"]],
                            WSDA_DataTable_2016[["ID", "ExctAcr", "Irrigtn"]],
                            WSDA_DataTable_2017[["ID", "ExctAcr", "Irrigtn"]],
                            WSDA_DataTable_2018[["ID", "ExctAcr", "Irrigtn"]]])

# %%
print (WSDA_DataTable.shape)
WSDA_DataTable = nc.filter_out_nonIrrigated(WSDA_DataTable)
print (WSDA_DataTable.shape)

# %%
(WSDA_DataTable[WSDA_DataTable.ExctAcr <= 10].ExctAcr.sum() / WSDA_DataTable.ExctAcr.sum())*100

# %%
WSDA_DataTable_2015 = nc.filter_out_nonIrrigated(WSDA_DataTable_2015)
WSDA_DataTable_2016 = nc.filter_out_nonIrrigated(WSDA_DataTable_2016)
WSDA_DataTable_2017 = nc.filter_out_nonIrrigated(WSDA_DataTable_2017)
WSDA_DataTable_2018 = nc.filter_out_nonIrrigated(WSDA_DataTable_2018)

# WSDA_DataTable_2015 = WSDA_DataTable_2015[WSDA_DataTable_2015.ExctAcr > 10]
# WSDA_DataTable_2016 = WSDA_DataTable_2016[WSDA_DataTable_2016.ExctAcr > 10]
# WSDA_DataTable_2017 = WSDA_DataTable_2017[WSDA_DataTable_2017.ExctAcr > 10]
# WSDA_DataTable_2018 = WSDA_DataTable_2018[WSDA_DataTable_2018.ExctAcr > 10]

# %%
WallaWalla = WSDA_DataTable_2015[WSDA_DataTable_2015.county == "Walla Walla"]

AdamsBenton = WSDA_DataTable_2016[WSDA_DataTable_2016.county.isin(["Adams", "Benton"])]

Grant = WSDA_DataTable_2017[WSDA_DataTable_2017.county.isin(["Grant"])]

FranklinYakima = WSDA_DataTable_2018[WSDA_DataTable_2018.county.isin(["Franklin", "Yakima"])]

WallaWalla.shape

# %%
WallaWalla = nc.filter_by_lastSurvey(WallaWalla, 2015)
AdamsBenton = nc.filter_by_lastSurvey(AdamsBenton, 2016)
Grant = nc.filter_by_lastSurvey(Grant, 2017)
FranklinYakima = nc.filter_by_lastSurvey(FranklinYakima, 2018)

WallaWalla.shape

# %%
len(WallaWalla) + len(AdamsBenton) + len(Grant) + len(FranklinYakima)

# %%
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

# %%
test20 = pd.merge(test20, meta, how="left", on="ID")

# %%
test20[test20.CropTyp=="mint"]

# %%

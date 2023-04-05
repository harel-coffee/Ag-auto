# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import os

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"

# %%
csv_files = [x for x in os.listdir(data_dir) if x.endswith(".csv")]

# %%
all_data=pd.DataFrame()
for a_file in csv_files:
    curr_file = pd.read_csv(data_dir+a_file)
    all_data=pd.concat([all_data, curr_file])
    
print (all_data.shape)
all_data.head(2)

# %%
all_data.dropna(subset=['county'], inplace=True) 
print (all_data.shape)

# %%
len(all_data.CropTyp.unique())

# %%
# import pprint
# pp = pprint.PrettyPrinter(indent=1)
# pp.pprint(sorted(all_data.CropTyp.unique()))

# %%

sorted(all_data.CropTyp.unique())

# %%
all_data.ExctAcr.sum()

# %%
sorted(all_data.county.unique())

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

# %%
all_testTrain = pd.concat([train80, test20])
all_testTrain.head(2)

# %%
all_testTrain.ID.isin(all_data.ID).sum()

# %%
all_testTrain.shape

# %%
all_data_subsetTest_train = all_data[all_data.ID.isin(list(all_testTrain.ID))]
all_data_subsetTest_train.shape

# %%
len(all_data_subsetTest_train.CropTyp.unique())

# %%
sorted(all_data_subsetTest_train.CropTyp.unique())

# %%
grass_hay = all_data[all_data.CropTyp=="alfalfa/grass hay"]
grass_hay=grass_hay[grass_hay.Acres>10]
grass_hay.shape

# %%
caneberry = all_data[all_data.CropTyp=="caneberry"]
caneberry=caneberry[caneberry.Acres>10]
caneberry.shape

# %%
walnut = all_data[all_data.CropTyp=="walnut"]
print (walnut.shape)
walnut=walnut[walnut.Acres>10]
print (walnut.shape)

# %%
pepper = all_data[all_data.CropTyp=="pepper"]
print (pepper.shape)
pepper=pepper[pepper.Acres>10]
print (pepper.shape)

# %%
all_data_subsetTest_train.shape

# %%
all_data_subsetTest_train.Acres.min()

# %%

# %%

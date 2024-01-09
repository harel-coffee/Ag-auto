# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from libtiff import TIFF;
# import gdal
from osgeo import gdal

# %% [markdown]
# ### Directories

# %%
range_data_dir = "/Users/hn/Documents/01_research_data/RangeLand/"
tiff_dir = range_data_dir + "Data/Matt_Reeves/"

# %% [markdown]
# #### files

# %%
HrbPctUnmaskedfile = "HrbPctUnmasked.tif"
HrbPctmasked = "HrbPctmasked.tif"

# %%

# %%
tif = TIFF.open(tiff_dir + 'HrbPctmasked.tif') # open tiff file in read mode
# read an image in the current TIFF directory as a numpy array
image = tif.read_image()

# %%
counter = 0
for image in tif.iter_images():
    counter += 1
    print (counter)
    pass

# %%
image.min()

# %%

dataset = gdal.Open(tiff_dir + 'HrbPctmasked.tif', gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()

# %%
for ii in range(1, dataset.RasterCount + 1):
    print (ii)

# %%
(array == image).sum()

# %%
array.shape[0]*array.shape[1]

# %%
print ((image<=-2).sum())
print ((image<=-10).sum())
print ((image<=-20).sum())
print ((image<=-50).sum())

# %%
(image==-128).sum()

# %%
(image==-1).sum()

# %%
import pandas as pd
A = pd.read_csv("/Users/hn/Documents/01_research_data/RangeLand/Data/Min_Data/statefips_annual_MODIS_NPP.csv")
A.head(2)

# %%

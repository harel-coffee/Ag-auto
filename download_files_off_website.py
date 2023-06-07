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
import requests
import pandas as pd

# %%
# url = 'https://www.facebook.com/favicon.ico'
# r = requests.get(url, allow_redirects=True)
# open('facebook.ico', 'wb').write(r.content)

# %%
# Change working directory if you want:

import os
print ("I am currently in " + os.getcwd())

data_dir = "/Users/hn/Documents/01_research_data/Reservoirs/"
os.chdir(data_dir)
print ("I am currently in " + os.getcwd())

# %%
resNames = pd.read_csv("/Users/hn/Documents/01_research_data/Reservoirs/Reservoirs.csv")
resNames = list(resNames.Name)
# ["Carters"]


# href = "https://nicholasinstitute.duke.edu/water/publications/creating-" + \
#          "data-service-us-army-corps-engineers-reservoirs/"

# href = "http://water.usace.army.mil/a2w/f?p=100:1:0:/"
# href = "http://www.mapbox.com/about/maps/"
# href = "https://nicholasinstitute.duke.edu/reservoir-data/"
href = "https://nicholasinstitute.duke.edu/reservoir-data/usace/data/daily/"
counter = 0
for reservoir in resNames:
    if (counter % 50 == 0):
        print ("counter is [{:.0f}].".format(counter))
    fileName = reservoir.replace(" ", "") +  ".csv" # remove spaces
    file_path =  href + fileName
    # print (file_path)
    r = requests.get(file_path, allow_redirects=True)
    open(fileName, 'wb').write(r.content)
    counter+=1

# %%
# Check

downloaded_by_code = pd.read_csv(data_dir + "/Carters.csv")
downloaded_manually = pd.read_csv("/Users/hn/Downloads/Carters.csv")
downloaded_manually.equals(downloaded_by_code)

# %%

# %%

# %%

# %%

# %%

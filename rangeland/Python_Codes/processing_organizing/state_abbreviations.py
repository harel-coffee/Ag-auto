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
import pandas as pd
import pickle

from datetime import datetime
import os, os.path, pickle, sys


# %%
state_to_abbrev = {"Alabama": "AL", 
                   "Alaska": "AK", 
                   "Arizona": "AZ",
                   "Arkansas": "AR", 
                   "California": "CA", 
                   "Colorado": "CO",
                   "Connecticut": "CT", 
                   "Delaware": "DE", 
                   "Florida": "FL",
                   "Georgia": "GA", 
                   "Hawaii": "HI", 
                   "Idaho": "ID",
                   "Illinois": "IL", 
                   "Indiana": "IN",
                   "Iowa": "IA",
                   "Kansas": "KS", 
                   "Kentucky": "KY",
                   "Louisiana": "LA",
                   "Maine": "ME",
                   "Maryland": "MD",
                   "Massachusetts": "MA",
                   "Michigan": "MI",
                   "Minnesota": "MN",
                   "Mississippi": "MS",
                   "Missouri": "MO",
                   "Montana": "MT",
                   "Nebraska": "NE",
                   "Nevada": "NV",
                   "New Hampshire": "NH",
                   "New Jersey": "NJ",
                   "New Mexico": "NM",
                   "New York": "NY",
                   "North Carolina": "NC",
                   "North Dakota": "ND",
                   "Ohio": "OH",
                   "Oklahoma": "OK",
                   "Oregon": "OR",
                   "Pennsylvania": "PA",
                   "Rhode Island": "RI",
                   "South Carolina": "SC",
                   "South Dakota": "SD",
                   "Tennessee": "TN",
                   "Texas": "TX",
                   "Utah": "UT",
                   "Vermont": "VT",
                   "Virginia": "VA",
                   "Washington": "WA",
                   "West Virginia": "WV",
                   "Wisconsin": "WI",
                   "Wyoming": "WY",
                   "District of Columbia": "DC",
                   "American Samoa": "AS",
                   "Guam": "GU",
                   "Northern Mariana Islands": "MP",
                   "Puerto Rico": "PR",
                   "United States Minor Outlying Islands": "UM",    
                   "U.S. Virgin Islands": "VI"}

states_abb_list = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                    'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                    'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                    'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

abb_2_full_dict = {'AK': 'Alaska',
                   'AL': 'Alabama',
                   'AR': 'Arkansas',    
                   'AZ': 'Arizona',
                   'CA': 'California',
                   'CO': 'Colorado',
                   'CT': 'Connecticut',
                   'DC': 'District of Columbia',
                   'DE': 'Delaware',
                   'FL': 'Florida',
                   'GA': 'Georgia',
                   'HI': 'Hawaii',
                   'IA': 'Iowa',
                   'ID': 'Idaho',
                   'IL': 'Illinois',
                   'IN': 'Indiana',
                   'KS': 'Kansas',
                   'KY': 'Kentucky',
                   'LA': 'Louisiana',
                   'MA': 'Massachusetts',
                   'MD': 'Maryland',
                   'ME': 'Maine',
                   'MI': 'Michigan',
                   'MN': 'Minnesota',
                   'MO': 'Missouri',
                   'MS': 'Mississippi',
                   'MT': 'Montana',
                   'NC': 'North Carolina',
                   'ND': 'North Dakota',
                   'NE': 'Nebraska',
                   'NH': 'New Hampshire',
                   'NJ': 'New Jersey',
                   'NM': 'New Mexico',
                   'NV': 'Nevada',
                   'NY': 'New York',
                   'OH': 'Ohio',
                   'OK': 'Oklahoma',
                   'OR': 'Oregon',
                   'PA': 'Pennsylvania',
                   'RI': 'Rhode Island',
                   'SC': 'South Carolina',
                   'SD': 'South Dakota',
                   'TN': 'Tennessee',
                   'TX': 'Texas',
                   'UT': 'Utah',
                   'VA': 'Virginia',
                   'VT': 'Vermont',
                   'WA': 'Washington',
                   'WI': 'Wisconsin',
                   'WV': 'West Virginia',
                   'WY': 'Wyoming'}


print (len(state_to_abbrev))
print (len(states_abb_list))
print (len(abb_2_full_dict))

# %%
SoI = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "Oregon",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Washington",
    "Wyoming",
]

# %%
import pickle
from datetime import datetime

data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

filename = param_dir + "state_abbreviations.sav"

export_ = {"full_2_abb" : state_to_abbrev,
           "states_abb_list" : states_abb_list,
           "abb_2_full" : abb_2_full_dict,
           "SoI" : SoI,
           "source_code": "state_abbreviations",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          }

pickle.dump(export_, open(filename, "wb"))

# %%

# %%

# %%

# %%

# %%

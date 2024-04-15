# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Get state adjacency
#
# Google found:
#
# - This [MATLAB page](https://blogs.mathworks.com/cleve/2018/09/03/graph-object-of-48-usa-states/#f9cd9f27-d722-45a6-ad11-06932ba6a70b) points to [web site of John Burkardt](https://people.sc.fsu.edu/~jburkardt/datasets/states/states.html)
#
# - This [GitHub repo](https://gist.github.com/rietta/4112447) points to [a wordpress page](https://writeonly.wordpress.com/2009/03/20/adjacency-list-of-states-of-the-united-states-us/)

# %%

# %%

# %%
import pandas as pd
import os
import requests;

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
# url = 'https://gist.githubusercontent.com/rietta/4112447/raw/d69f1b6aa855de17131a85090a9ac5f2fbd805ae/'
# url = url + "us_states_adj.txt"
# response = requests.get(url) # Generate response object
# text = response.text         # Return the HTML of webpage as string
# text.splitlines()


# %%
gitHubStateAdj = open(param_dir + "gitHubStateAdj.txt", "r")
gitHubStateAdj = gitHubStateAdj.read()
gitHubStateAdj = gitHubStateAdj.splitlines()
gitHubStateAdj[:4]

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print (f"{len(state_fips_SoI) = }")
print (f"{state_fips.shape = }")
state_fips_SoI.head(2)

# %%
adj_df = pd.DataFrame(columns=list(state_fips.state.values), index=list(state_fips.state.values))
adj_df.fillna(0, inplace=True)
adj_df.head(2)

# %%
for entry in gitHubStateAdj:
    entry_list = entry.split(",")
    curr_state = entry_list.pop(0)
    for a_neighbor in entry_list:
        adj_df.loc[curr_state, a_neighbor] = 1
        adj_df.loc[a_neighbor, curr_state] = 1

# %%
adj_df.head(2)

# %%
adj_df_fips = adj_df.copy()
adj_df_fips.columns = state_fips.state_fips.values
adj_df_fips.index = state_fips.state_fips.values

adj_df_fips.head(2)

# %%
print (len(state_fips_SoI))
state_fips_SoI.head(2)

# %% [markdown]
# ## Subset to states of interest

# %%
adj_df_SoI = adj_df[list(state_fips_SoI.state)].copy()
adj_df_SoI = adj_df_SoI.loc[list(state_fips_SoI.state)]
print (adj_df_SoI.shape)
adj_df_SoI.head(2)

# %%
adj_df_fips.shape

# %%
adj_df_fips_SoI = adj_df_fips[list(state_fips_SoI.state_fips)].copy()
adj_df_fips_SoI = adj_df_fips_SoI.loc[list(state_fips_SoI.state_fips)]
print (adj_df_fips_SoI.shape)
adj_df_fips_SoI.head(2)

# %%
import numpy as np
np.diag(adj_df_SoI)

# %% [markdown]
# # Spectral normalized

# %%
eigenvalues, eigenvectors = np.linalg.eig(adj_df)
print (eigenvalues.max().round(2))
adj_df_spectralNormalized = adj_df / max(eigenvalues)

eigenvalues, _ = np.linalg.eig(adj_df_spectralNormalized)
eigenvalues.max().round(2)

# %%
eigenvalues, eigenvectors = np.linalg.eig(adj_df_SoI)
adj_df_SoI_spectralNormalized = adj_df_SoI / max(eigenvalues)

eigenvalues, eigenvectors = np.linalg.eig(adj_df_fips)
adj_df_fips_spectralNormalized = adj_df_fips / max(eigenvalues)

eigenvalues, eigenvectors = np.linalg.eig(adj_df_fips_SoI)
adj_df_fips_SoI_spectralNormalized = adj_df_fips_SoI / max(eigenvalues)

# %% [markdown]
# ### Row normalized

# %%
adj_df_rowNormalized = adj_df.div(adj_df.sum(axis=0), axis=0)
adj_df_SoI_rowNormalized = adj_df_SoI.div(adj_df_SoI.sum(axis=0), axis=0)
adj_df_fips_rowNormalized = adj_df_fips.div(adj_df_fips.sum(axis=0), axis=0)
adj_df_fips_SoI_rowNormalized = adj_df_fips_SoI.div(adj_df_fips_SoI.sum(axis=0), axis=0)

# %%

# %%
neighbors_dict_abb = {}

for entry in gitHubStateAdj:
    curr_list = entry.split(",")
    curr_state = curr_list.pop(0)
    neighbors_dict_abb[curr_state] = curr_list 

    
df = state_fips.copy()
df.set_index('state', inplace=True)
df.head(2)

neighbors_dict_fips = {}
for entry in gitHubStateAdj:
    curr_list = entry.split(",")
    curr_state = curr_list.pop(0)
    neighbors_dict_fips[df.loc[curr_state]["state_fips"]] = df.loc[curr_list]["state_fips"].to_list() 

# %%

# %%

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "state_adj_dfs.sav"

export_ = {"adj_df": adj_df,
           "adj_df_SoI" : adj_df_SoI,
           "adj_df_fips": adj_df_fips,
           "adj_df_fips_SoI": adj_df_fips_SoI,
           
           "adj_df_spectralNormalized": adj_df_spectralNormalized,
           "adj_df_SoI_spectralNormalized" : adj_df_SoI_spectralNormalized,
           "adj_df_fips_spectralNormalized": adj_df_fips_spectralNormalized,
           "adj_df_fips_SoI_spectralNormalized": adj_df_fips_SoI_spectralNormalized,
           
           
           "adj_df_rowNormalized" : adj_df_rowNormalized,
           "adj_df_SoI_rowNormalized" : adj_df_SoI_rowNormalized,
           "adj_df_fips_rowNormalized" : adj_df_fips_rowNormalized,
           "adj_df_fips_SoI_rowNormalized" : adj_df_fips_SoI_rowNormalized,
           
           "neighbors_dict_abb" : neighbors_dict_abb,
           "neighbors_dict_fips" : neighbors_dict_fips,

           "source_code" : "04_11_2024_stateAdjacency",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
neighbors_dict_abb

# %%

# %%
neighbors_dict_fips

# %%
from libpysal.weights import W

# %%
weights = {}
for a_key in neighbors_dict_fips.keys():
    weights[a_key] = [1] * len(neighbors_dict_fips[a_key])

# %%
adj_pysal_weights_ = W(neighbors_dict_fips, weights)

# %%
adj_pysal_weights_.histogram

# %%

# %%

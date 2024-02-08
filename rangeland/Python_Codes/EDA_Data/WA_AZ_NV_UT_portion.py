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

# %%
import pandas as pd

from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter


# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

plots_dir = data_dir_base + "plots/"

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict["SoI"]
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
invent_tall = pd.read_pickle(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_tall.sav")
invent_tall = invent_tall["CATINV_annual_tall"]

invent_tall = invent_tall[invent_tall.state.isin(SoI_abb)]

invent_tall.reset_index(drop=True, inplace=True)
invent_tall.head(2)

# %%
US_inventory = invent_tall.groupby(["year"])["inventory"].sum().reset_index()
US_inventory.rename(columns={"inventory": "US_inventory"}, inplace=True)
US_inventory.head(2)

# %%
invent_tall = pd.merge(invent_tall, US_inventory, on=["year"], how="left")
invent_tall.head(2)

# %%
invent_tall["state_perc"] = (100 * invent_tall["inventory"]) / invent_tall[
    "US_inventory"
]
invent_tall.head(2)

# %%
invent_tall[invent_tall.state == "WA"]

# %%
invent_tall[invent_tall.state == "TX"]

# %%
chosen_states = ["TX", "AL", "LA", "CA", "WA", "UT", "NV", "AZ"]

fig, axs = plt.subplots(
    2, 1, figsize=(10, 4), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05}
)

axs[0].grid(axis="y", which="both")
axs[1].grid(axis="y", which="both")
#####################################################################################
ii = 0
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "dodgerblue"
axs[0].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 1
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "red"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 2
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "k"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 3
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 4
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 5
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 6
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 7
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################


axs[1].set_ylabel("inventory (% of US)")
axs[1].legend(loc="best")

axs[0].set_ylabel("inventory (% of US)")
axs[0].legend(loc="best")


# plt.tight_layout()
plots_dir = data_dir_base + "plots/"
fig_name = plots_dir + "Tonsor_states.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fig, axs = plt.subplots(
    1, 1, figsize=(10, 2), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05}
)
axs.grid(axis="y", which="both")
#####################################################################################
state_ = sorted(invent_tall.state.unique())[28]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "dodgerblue"
axs.plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
axs.set_ylabel("inventory (% of US)")
axs.legend(loc="best")

# plots_dir = data_dir_base + "plots/"
# fig_name = plots_dir + "increase_states.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%

# %%
invent_tall = invent_tall[invent_tall.year >= 1992]

chosen_states = ["TX", "AL", "LA", "CA", "WA", "UT", "NV", "AZ"]

fig, axs = plt.subplots(
    2, 1, figsize=(10, 4), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05}
)

axs[0].grid(axis="y", which="both")
axs[1].grid(axis="y", which="both")
#####################################################################################
ii = 0
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "dodgerblue"
axs[0].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 1
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "red"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 2
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "k"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 3
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 4
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 5
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 6
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################
ii = 7
state_ = chosen_states[ii]
B = invent_tall[invent_tall.state == state_].copy()
color_ = "c"
axs[1].plot(B.year, B.state_perc, linewidth=3, label=state_)
#####################################################################################


axs[1].set_ylabel("inventory (% of US)")
axs[1].legend(loc="best")

axs[0].set_ylabel("inventory (% of US)")
axs[0].legend(loc="best")


# plt.tight_layout()
plots_dir = data_dir_base + "plots/"
fig_name = plots_dir + "Tonsor_states_1992.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

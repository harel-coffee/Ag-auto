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
import pickle

# %%
range_dir_ = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
weather_dir_ = range_dir_ + "reOrganized/seasonal_variables/02_merged_mean_over_county/"
mike_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/Mike_Oct25/"

# %%
file_name = "countyMean_seasonalVars_wFips.sav"
countyMean_seasonalVars = pickle.load(open(weather_dir_ + file_name, "rb"))


wide_seasonal_vars_cntyMean_wFips = pickle.load(open(weather_dir_ + "wide_seasonal_vars_cntyMean_wFips.sav", "rb"))

# %%
countyMean_seasonalVars.keys()

# %%
wide_seasonal_vars_cntyMean_wFips.keys()

# %%
wide_seasonal_vars_cntyMean_wFips["wide_seasonal_vars_cntyMean_wFips"].head(2)

# %%
countyMean_seasonalVars["countyMean_seasonalVars_wFips"].head(2)

# %%
out_name = mike_dir + "countyMean_seasonalVars_wFips.csv"
countyMean_seasonalVars["countyMean_seasonalVars_wFips"].to_csv(out_name, index = False)

# %%
out_name = mike_dir + "wide_seasonal_vars_cntyMean_wFips.csv"
wide_seasonal_vars_cntyMean_wFips["wide_seasonal_vars_cntyMean_wFips"].to_csv(out_name, index = False)

# %%
min_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/Min_Data/"

# %%
county_annual_MODIS_GPP = pd.read_csv(min_dir + "county_annual_MODIS_GPP.csv")
county_annual_MODIS_GPP.head(2)

# %%
county_annual_MODIS_NPP = pd.read_csv(min_dir + "county_annual_MODIS_NPP.csv")
county_annual_MODIS_NPP.head(2)

# %%
county_annual_productivity = pd.read_csv(min_dir + "county_annual_productivity.csv")
county_annual_productivity.head(2)

# %%
prod_county = county_annual_productivity.county.unique()
NPP_county =  county_annual_MODIS_NPP.county.unique()
GPP_county =  county_annual_MODIS_GPP.county.unique()

# %%
print (f"{len(prod_county) = }")
print (f"{len(NPP_county) = }")
print (f"{len(GPP_county) = }")

# %%
county_annual = county_annual_MODIS_GPP[["year", "county", "GPP"]].merge(\
                county_annual_MODIS_NPP[["year", "county", "NPP"]], how="left", on=["year", "county"])
print (f"{county_annual.shape = }")
print (f"{county_annual_MODIS_GPP.shape = }")
print (f"{county_annual_MODIS_NPP.shape = }")

# %%
county_annual.head(2)

# %%
county_annual_outer = county_annual_MODIS_NPP[["year", "county", "NPP"]].merge(\
                      county_annual_MODIS_GPP[["year", "county", "GPP"]], how="outer", on=["year", "county"])
print (f"{county_annual_outer.shape = }")
county_annual_outer.head(2)

# %%
county_annual_outer.equals(county_annual)

# %%
county_annual_outer = county_annual_outer[["year", "county", "NPP", "GPP"]]
county_annual = county_annual[["year", "county", "NPP", "GPP"]]

county_annual_outer.sort_values(by=["year", "county"], inplace=True)
county_annual.sort_values(by=["year", "county"], inplace=True)

county_annual_outer.reset_index(drop=True, inplace=True)
county_annual.reset_index(drop=True, inplace=True)

# %%

# %%
county_annual_outer[["year", "county", "NPP", "GPP"]].equals(county_annual[["year", "county", "NPP", "GPP"]])

# %%
county_annual_outer_all = county_annual_outer.merge(county_annual_productivity, 
                                                    how="outer", on=["year", "county"])
county_annual_outer_all.head(2)

# %%
print (f"{county_annual_outer_all.shape = }")
print (f"{min(county_annual_outer.year) = }")
print (f"{max(county_annual_outer.year) = }")
print ()
print (f"{min(county_annual_productivity.year) = }")
print (f"{max(county_annual_productivity.year) = }")
print ()
print (f"{min(county_annual_outer_all.year) = }")
print (f"{max(county_annual_outer_all.year) = }")
print ()

# %%
out_name = mike_dir + "county_annual_NPPGPPProductivity.csv"
county_annual_outer_all.to_csv(out_name, index = False)

# %%

# %%
county_monthly_AVHRR_NDVI = pd.read_csv(min_dir + "county_monthly_AVHRR_NDVI.csv")
county_monthly_GIMMS_NDVI = pd.read_csv(min_dir + "county_monthly_GIMMS_NDVI.csv")
county_monthly_MODIS_GPP = pd.read_csv(min_dir + "county_monthly_MODIS_GPP.csv")
county_monthly_MODIS_LAI = pd.read_csv(min_dir + "county_monthly_MODIS_LAI.csv")
county_monthly_MODIS_NDVI = pd.read_csv(min_dir + "county_monthly_GIMMS_NDVI.csv")
county_monthly_MODIS_PsnNet = pd.read_csv(min_dir + "county_monthly_MODIS_PsnNet.csv")

# %%
print (f"{county_monthly_AVHRR_NDVI.shape=}")
print (f"{county_monthly_GIMMS_NDVI.shape=}")
print (f"{county_monthly_MODIS_NDVI.shape=}")
print ()

print (f"{county_monthly_MODIS_GPP.shape=}")
print (f"{county_monthly_MODIS_PsnNet.shape=}")
print ()
print (f"{county_monthly_MODIS_LAI.shape=}")

# %%
county_monthly_AVHRR_NDVI.head(2)

# %%
county_monthly_GIMMS_NDVI.head(2)

# %%
county_monthly_AVHRR_NDVI.rename(columns={"NDVI": "AVHRR_NDVI"}, inplace=True)
county_monthly_GIMMS_NDVI.rename(columns={"NDVI": "GIMMS_NDVI"}, inplace=True)

# %%
county_monthly_MODIS_NDVI.head(2)

# %%
county_monthly_MODIS_NDVI.rename(columns={"NDVI": "MODIS_NDVI"}, inplace=True)
county_monthly_MODIS_NDVI.head(2)

# %%
county_monthly_MODIS_GPP.head(2)

# %%
county_monthly_MODIS_GPP.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
county_monthly_MODIS_GPP.head(2)

# %%
county_monthly_MODIS_PsnNet.head(2)

# %%
county_monthly_MODIS_PsnNet.rename(columns={"PsnNet": "MODIS_PsnNet"}, inplace=True)
county_monthly_MODIS_PsnNet.head(2)

# %%
county_monthly_MODIS_LAI.head(2)

# %%
county_monthly_MODIS_LAI.rename(columns={"LAI": "MODIS_LAI"}, inplace=True)
county_monthly_MODIS_LAI.head(2)

# %%
county_monthly_outer = county_monthly_AVHRR_NDVI.merge(\
                        county_monthly_GIMMS_NDVI, how="outer", on=["year", "month", "county"])


# %%
county_monthly_outer = county_monthly_outer.merge(\
                        county_monthly_MODIS_NDVI, how="outer", on=["year", "month", "county"])

# %%
county_monthly_outer = county_monthly_outer.merge(\
                        county_monthly_MODIS_GPP, how="outer", on=["year", "month", "county"])

# %%
county_monthly_outer = county_monthly_outer.merge(\
                        county_monthly_MODIS_PsnNet, how="outer", on=["year", "month", "county"])

# %%
county_monthly_outer = county_monthly_outer.merge(\
                        county_monthly_MODIS_LAI, how="outer", on=["year", "month", "county"])

# %%
county_monthly_outer.head(2)

# %%
county_monthly_outer.shape

# %%
county_monthly_outer.reset_index(drop=True, inplace=True)

# %%
out_name = mike_dir + "county_monthly_productivity.csv"
county_monthly_outer.to_csv(out_name, index = False)

# %%

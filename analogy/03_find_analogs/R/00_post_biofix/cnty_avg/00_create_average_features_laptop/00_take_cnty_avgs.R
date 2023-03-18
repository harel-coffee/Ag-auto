rm(list=ls())
library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats)

options(digit=9)
options(digits=9)

#_____________________________________________________________________________
# Directories:
# 
main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/00_databases/"
param_in <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

#_____________________________________________________________________________
# Read data
#
CDD_precip_rcp45 <- data.table(readRDS(paste0(main_in, "CDD_precip_rcp45.rds")))
CDD_precip_rcp85 <- data.table(readRDS(paste0(main_in, "CDD_precip_rcp85.rds")))
hist_CDD_precip <- data.table(readRDS(paste0(main_in, "hist_CDD_precip.rds")))


#_____________________________________________________________________________
# Read parameters to add fips, and compute averages over counties.
#
fips_info <- read.csv(paste0(param_in, "Min_fips_st_county_location.csv"), as.is=T, header=T)
all_us_1300_cty <- read.csv(paste0(param_in, "all_us_1300_county_fips_locations.csv"), as.is=T, header=T)
local_county_fips <- read.csv(paste0(param_in, "local_county_fips.csv"), as.is=T, header=T)

#_____________________________________________________________________________
# Get rid of the 8 locations
#

CDD_precip_rcp45 <- CDD_precip_rcp45 %>% filter(location %in% hist_CDD_precip$location) %>% data.table()
CDD_precip_rcp85 <- CDD_precip_rcp85 %>% filter(location %in% hist_CDD_precip$location) %>% data.table()

CDD_precip_rcp45 <- merge(CDD_precip_rcp45, fips_info, by = "location", all.x = T)
CDD_precip_rcp85 <- merge(CDD_precip_rcp85, fips_info, by = "location", all.x = T)
hist_CDD_precip <- merge(hist_CDD_precip, fips_info, by = "location", all.x = T)

#_____________________________________________________________________________
# Compute averages over counties.
#

CDD_precip_rcp45 = CDD_precip_rcp45[, .(mean_CumDDinF_Aug23 = mean(CumDDinF_Aug23), 
                                        mean_yearly_precip = mean(yearly_precip)), 
                                      by = c("year", "model", "fips", "st_county")]

CDD_precip_rcp85 = CDD_precip_rcp85[, .(mean_CumDDinF_Aug23 = mean(CumDDinF_Aug23), 
                                        mean_yearly_precip = mean(yearly_precip)), 
                                      by = c("year", "model", "fips", "st_county")]


hist_CDD_precip = hist_CDD_precip[, .(mean_CumDDinF_Aug23 = mean(CumDDinF_Aug23), 
                                      mean_yearly_precip = mean(yearly_precip)), 
                                      by = c("year", "model", "fips", "st_county")]


saveRDS(CDD_precip_rcp45, paste0(main_in, "cnty_avg_feat_45.rds"))
saveRDS(CDD_precip_rcp85, paste0(main_in, "cnty_avg_feat_85.rds"))
saveRDS(hist_CDD_precip, paste0(main_in, "cnty_avg_feat_hist.rds"))

#_____________________________________________________________________________
# Break down to (county, model) level
#
all_models <- sort(unique(CDD_precip_rcp45$model))
all_counties <- sort(unique(CDD_precip_rcp45$st_county))
emissions <- c("rcp45", "rcp85")

for (emission  in emissions){
  if (emission == "rcp45"){ data <- CDD_precip_rcp45} else {data <- CDD_precip_rcp85}
  for (a_model in all_models){
    out_dir <- paste0(main_in, "county_averages", "/", emission, "/", a_model, "/")
    if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }
    for (county in all_counties){
      curr_data <- data %>% filter(st_county == county & model == a_model)
      saveRDS(curr_data, paste0(out_dir, "feat_", gsub(" ", "_", county),".rds"))
    }
  }
}





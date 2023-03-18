######################################################################
rm(list=ls())

library(lubridate)
library(MASS)
library(purrr)
library(maps)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/analogy/core_analog.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/analogy/core_analog_plots.R"
source(source_path_1)
source(source_path_2)

# library(swfscMisc) has na.count(.) in it

options(digit=9)
options(digits=9)
######################################################################
####
####           global Files
####
######################################################################
param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"
new_param_dir <- paste0(param_dir, "new_params_for_analog/")

min_grids <- 5
hist_grid_count <- read.csv(paste0(new_param_dir, "cod_moth_hist_grid_count_within_counties.csv"),
                            header=T, as.is=T) %>%                            
                            filter(grid_count >= min_grids) %>%
                            data.table()

hist_grid_count$st_county <- paste(hist_grid_count$state, hist_grid_count$county, sep="_" )
hist_grid_count <- within(hist_grid_count, remove(state, county))

f_loc_fips_st_cnty <- "local_county_fips.csv"
h_loc_fips_st_cnty <- "all_us_1300_county_fips_locations.csv"

f_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, f_loc_fips_st_cnty), header=T, sep=",", as.is=T))
h_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, h_loc_fips_st_cnty), header=T, sep=",", as.is=T))

f_loc_fips_st_cnty <- get_286_locs(f_loc_fips_st_cnty, h_loc_fips_st_cnty)

local_fips <- unique(f_loc_fips_st_cnty$fips)
 
######## Filter h_loc_fips_st_cnty, so, it only includes
######## counties with minimum number of grids in it. 
######## Doing so in this step may prevent slipping 
######## not qualified counties in calculations:
######## Doing so, takes us from 1293 locations to 1175. (118 locations are gone)

h_loc_fips_st_cnty <- h_loc_fips_st_cnty %>% filter(fips %in% hist_grid_count$fips)

######################################################
# 
# To extract names of counties associated with each unique fips
# 
Min_fips_st_county <- data.table(read.csv(paste0(param_dir, "Min_fips_st_county_location.csv"), 
                                 header=T, sep=",", as.is=T)) 
Min_fips_st_county <- within(Min_fips_st_county, remove(location))
Min_fips_st_county <- unique(Min_fips_st_county)

######################################################################

model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
time_p <- c("2026_2050", "2051_2075", "2076_2095")
emissions <- c("rcp45", "rcp85")
sigma_bds <- c(1, 2)

######################################################################
####
####         Set up directories
####
######################################################################

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/"
data_sub_dirs <- c("w_precip_rcp45/", "w_precip_rcp85/")

sub_dir <- data_sub_dirs[1]
sigma_bd <- 2
target_fip <- 53021
model_n <- model_names[6]

################################################################################
################################################################################
iin_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/00_databases/"
F85_feat <- data.table(readRDS(paste0(iin_dir, "CDD_precip_rcp85.rds")))
hist_feat <- data.table(readRDS(paste0(iin_dir, "hist_CDD_precip.rds")))


## Filter Canyon county, bcc model, 2026-2059 time period.
F85_feat <- merge(F85_feat, f_loc_fips_st_cnty, by="location", all.x=T)
F85_feat <- na.omit(F85_feat) # we had dropped the 9 locations to get 286 locs
F85_feat <- F85_feat %>% 
            filter(model == "bcc-csm1-1-m" & st_county == "ID_Canyon" & year <= 2050 & year > 2025) %>% 
            data.table()

## Filter Historical Walla Walla

hist_feat <- merge(hist_feat, h_loc_fips_st_cnty, by="location", all.x=T)
hist_feat <- na.omit(hist_feat)
hist_feat <- hist_feat %>% filter(st_county == "WA_Walla Walla")

################################################################################
#
# Plot the contours of distributions
#

# Matrix of future. (kde2d does not work with data table)

MF <- as.matrix(F85_feat[, c("CumDDinF_Aug23", "yearly_precip")])
F85_feat_kde <- kde2d(MF[, 1], MF[, 2], n = 50)

HF <- as.matrix(hist_feat[, c("CumDDinF_Aug23", "yearly_precip")])
hist_feat_kde <- kde2d(HF[, 1], HF[, 2], n = 50)

image(F85_feat_kde)
contour(F85_feat_kde, add = TRUE) # from base graphics package

image(hist_feat_kde)
contour(hist_feat_kde, add = TRUE)
################################################################################
#
# Plot the contours of distributions ((GGPLOT))
#

future_hist <- rbind(F85_feat, hist_feat)

 ggplot(future_hist, aes(x = CumDDinF_Aug23, y = yearly_precip)) + 
 # geom_point() + 
 # geom_density_2d() + 
 stat_density_2d(aes(fill = stat(level), colour = model), geom = "raster", contour = FALSE)



ggplot(future_hist, aes(x = CumDDinF_Aug23, y = yearly_precip)) + 
# geom_point() + 
# geom_density_2d() + 
stat_density_2d(aes(fill = stat(level), colour = model), alpha=.5, 
                contour = TRUE, geom = "polygon") + 
theme(plot.title = element_text(size=20, face="bold"),
                                plot.margin = unit(c(t=.5, r=0.1, b= -2, l=0.1), "cm"),
                                legend.title = element_blank(),
                                legend.position = "bottom",
                                legend.key.size = unit(3.3, "line"),
                                legend.text = element_text(size=20, face="bold"),
                                legend.margin = margin(t=.5, r=0, b=1, l=0, unit = 'cm'),
                                axis.text.x = element_blank(),
                                axis.text.y = element_blank(),
                                axis.ticks.x = element_blank(),
                                axis.ticks.y = element_blank(),
                                axis.title.x = element_blank(),
                                axis.title.y = element_blank())


ggplot(future_hist, aes(x = CumDDinF_Aug23, y = yearly_precip)) + 
# geom_point() + 
# geom_density_2d() + 
geom_density_2d(aes(fill = stat(level), colour = model), contour = TRUE, geom = "polygon")



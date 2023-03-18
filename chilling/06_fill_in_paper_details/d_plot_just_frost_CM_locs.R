# We could/should create two sets of data for each 
# (of the first two) 
# scenarios above or, we can take care of NAs
# -introduced to data by merging frost and bloom-
# in the plotting functions?
#####################################
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
# basee <- "/Users/hn/Documents/GitHub/Ag/bloom/"
# source_1 = paste0(basee, "bloom_core.R")
# source_2 = paste0(basee, "/bloom_plot_core.R")
# source(source_1)
# source(source_2)

# in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/bloom/"
# param_dir <- paste0(basee, "parameters/")
# plot_base_dir <- in_dir
############################################################
###
###             Aeolus source
###
############################################################
source_dir <- "/home/hnoorazar/bloom_codes/"
source_1 <- paste0(source_dir, "bloom_core.R")
source_2 <- paste0(source_dir, "bloom_plot_core.R")
source(source_1)
source(source_2)
############################################################
###
###             Aeolus Directories
###

base <- "/data/hydro/users/Hossein/bloom/"
in_dir <- paste0(base, "03_merge_02_Step/")
param_dir <- paste0(source_dir, "parameters/")
plot_base_dir <- paste0(base, "04_bloom_vs_frost_plot/CM_locs/")
#############################################################
###
###               Read data off the disk
###
#############################################################
#
# for sake of iteration
#
location_ls <- read.csv(file = paste0(param_dir, "cm_locs.csv"), 
                        header=TRUE, as.is=TRUE)

#
# convert chill day of year to readable day of year!
#
chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), 
                          as.is=TRUE)
#############################################################

frost <- readRDS(paste0(in_dir, "first_frost.rds"))

# for some reason frost has 2015-2016 in it!
# frost <- frost %>% filter(chill_season != "chill_2015-2016")
# frost <- frost %>% filter(chill_season != "chill_2098-2099")
# saveRDS(frost, paste0(in_dir, "first_frost.rds"))
#
# original sept_summary_comp has some columns we do not want!
# time periods are separate, observed and modeled hist do not have
# two separate emissions in them, so:
#
# thresh <- change_thresh_4_bloom(thresh, location_ls)
# thresh <- thresh %>% filter(chill_season != "chill_2098-2099")
# thresh <- convert_zero_to_365(thresh)
# saveRDS(thresh, paste0(in_dir, "sept_summary_comp.rds"))

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

frost <- pick_obs_and_F(frost)
#############################################################
#
#              clean up each data table
#
#############################################################
frost <- within(frost, remove(tmin))
# for sake of iteration:
frost <- add_location(frost)

suppressWarnings({frost <- within(frost, remove(lat, long))})

print(length(unique(frost$chill_season)))
print(length(unique(frost$location)))
print(length(unique(frost$model)))
#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")

###############################################################
em <- emissions[2]
loc <- location_ls$location[1]

start_time <- Sys.time()
frost_plot_dir <- paste0(plot_base_dir, 
                         "frost_plots/")

if (dir.exists(frost_plot_dir) == F) {
  dir.create(path = frost_plot_dir, recursive = T)
}

for (loc in location_ls$location){
  curr_lat <- substr(unique(loc), 1, 8)
  curr_long <- substr(unique(loc), 11, 19)
  for (em in emissions){
    curr_frost <- frost %>% 
                  filter(location==loc & emission==em) %>% 
                  data.table()

    #############################################################
    ##
    ##             Frost plot here
    ##
    frost_plt <- cloudy_frost(d1 = curr_frost, 
                              colname = "chill_dayofyear", 
                              fil = "first frost") + 
                 ggtitle(lab=paste0("first frost shift at (",
                                    curr_lat, " N, ",
                                    curr_long, " W), ",
                                    em))
    ggsave(plot = frost_plt,
           filename = paste0(loc, "_", gsub(" ", "_", em), ".png"), 
           width=8, height=4, units = "in", 
           dpi=400, device = "png",
           path=frost_plot_dir)
  }
}

print(Sys.time() - start_time)



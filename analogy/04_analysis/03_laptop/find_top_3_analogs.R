######################################################################
#rm(list=ls())

library(lubridate)
library(purrr)
library(maps)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

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
                            header=T, as.is=T)  %>%                            
                            filter(grid_count >= min_grids) %>%
                            data.table()

hist_grid_count$st_county <- paste(hist_grid_count$state, hist_grid_count$county, sep="_" )
hist_grid_count <- within(hist_grid_count, remove(state, county))

###
###

f_loc_fips_st_cnty <- "local_county_fips.csv"
h_loc_fips_st_cnty <- "all_us_1300_county_fips_locations.csv"

f_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, f_loc_fips_st_cnty), header=T, sep=",", as.is=T))
h_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, h_loc_fips_st_cnty), header=T, sep=",", as.is=T))

f_loc_fips_st_cnty <- f_loc_fips_st_cnty %>% 
                      filter(location %in% h_loc_fips_st_cnty$location) %>% 
                      data.tabale()

local_fips <- unique(f_loc_fips_st_cnty$fips)
 
# f_grid_count <- f_loc_fips_st_cnty %>%
#                 group_by(fips, st_county) %>%
#                 transmute(grid_count = n_distinct(location)) %>%
#                 unique() %>%
#                 data.table()

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

######################################################

######################################################

model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
time_p <- c("2026_2050", "2051_2075", "2076_2095")
emissions <- c("rcp45", "rcp85")

######################################################################
####
####         Set up directories
####
######################################################################

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/"
data_sub_dirs <- c("w_precip_rcp45/", "w_precip_rcp85/")

sub_dir <- data_sub_dirs[1]

for (sub_dir in data_sub_dirs){
  emission <- substr(unlist(strsplit(sub_dir, "_"))[3], 1, 5)
  # name_extension <- paste0(strsplit(sub_dir, "_")[[1]][1], 
  #                          "_", strsplit(sub_dir, "_")[[1]][3], 
  #                          "_", emission)
  name_extension <- paste0(strsplit(sub_dir, "_")[[1]][1], 
                           "_", strsplit(sub_dir, "_")[[1]][2], 
                           "_", emission)
  sigma_bds <- c(1, 2)

  ######################################################################
  sigma_bd <- sigma_bds[1]
  target_fip <- local_fips[1]
  model_n <- model_names[1]

  for (sigma_bd in sigma_bds){
    data_dir <- paste0(main_in, sigma_bd, "_sigma/", sub_dir)
    # print(data_dir)
    top_3_dt <- data.table()

    for (target_fip in local_fips){
      for(model_n in model_names){
        # read the data of the three time periods
        top_m_dt = data.table(future_fip = c(target_fip, target_fip, target_fip),
                              model = c(model_n, model_n, model_n),
                              time_period = c("F1", "F2", "F3"),
                              emission = c(emission, emission, emission),
                              top_1_fip = c(666, 666, 666),
                              top_2_fip = c(666, 666, 666),
                              top_3_fip = c(666, 666, 666)
                              )

        analog_file_name_F1 <- paste("analog", model_n, emission, time_p[1], sep="_")
        analog_dat_F1 <- data.table(readRDS(paste0(data_dir, analog_file_name_F1, ".rds")))

        analog_file_name_F2 <- paste("analog", model_n, emission, time_p[2], sep="_")
        analog_dat_F2 <- data.table(readRDS(paste0(data_dir, analog_file_name_F2, ".rds")))

        analog_file_name_F3 <- paste("analog", model_n, emission, time_p[3], sep="_")
        analog_dat_F3 <- data.table(readRDS(paste0(data_dir, analog_file_name_F3, ".rds")))
        #############################################################################

        #############################################################################
        #
        #   Filter out counties with less than "min_grids" grids in them.
        #  *** Let us remove counties with historical #grids < min_grids from the map ***
        
        # This automatically removes not_analogs!
        analog_dat_F1 <- analog_dat_F1 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips)
        analog_dat_F2 <- analog_dat_F2 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips)
        analog_dat_F3 <- analog_dat_F3 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips)

        #############################################################################

        # filter the location of interest:
        analog_dat_F1 <- analog_dat_F1 %>% filter(query_county == target_fip)
        analog_dat_F2 <- analog_dat_F2 %>% filter(query_county == target_fip)
        analog_dat_F3 <- analog_dat_F3 %>% filter(query_county == target_fip)
     
        # remove the stats of not_analog pieces
        analog_dat_F1 <- analog_dat_F1 %>% filter(analog_NNs_county != "no_analog")
        analog_dat_F2 <- analog_dat_F2 %>% filter(analog_NNs_county != "no_analog")
        analog_dat_F3 <- analog_dat_F3 %>% filter(analog_NNs_county != "no_analog")

        #############################################################################
        # 
        # Determine how many years are in future data frames, to be used
        # for standardization of maps and data needed for donuts.
        
        f_years_F1 = 1 + as.numeric(unlist(strsplit(time_p[1], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[1], "_")))[1]

        f_years_F2 = 1 + as.numeric(unlist(strsplit(time_p[2], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[2], "_")))[1]

        f_years_F3 = 1 + as.numeric(unlist(strsplit(time_p[3], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[3], "_")))[1]

        #############################################################################
        #############################################################################
        #
        # *** Standardize the freq. using # of grids in each county (historical data) ***
        # hist_counties_with_more_10 has the fips and count of grids within each county.
        # But then we talked in her office and we changed the denominator to all possible
        # pairwise combinations, so, fractions are between 0 and 1.
        # This should be done just for maps.
        analog_dat_F1 <- standardize_by_all_pairs(analog_dt=analog_dat_F1, 
                                                  f_fips_dt=f_loc_fips_st_cnty, 
                                                  h_fips_dt=h_loc_fips_st_cnty, 
                                                  f_years=f_years_F1, h_years=37)
        
        analog_dat_F2 <- standardize_by_all_pairs(analog_dt=analog_dat_F2, 
                                                  f_fips_dt=f_loc_fips_st_cnty, 
                                                  h_fips_dt=h_loc_fips_st_cnty, 
                                                  f_years=f_years_F2, h_years=37)
        
        analog_dat_F3 <- standardize_by_all_pairs(analog_dt=analog_dat_F3, 
                                                  f_fips_dt=f_loc_fips_st_cnty, 
                                                  h_fips_dt=h_loc_fips_st_cnty, 
                                                  f_years=f_years_F3, h_years=37)

        analog_dat_F1 <- analog_dat_F1[[1]]
        analog_dat_F2 <- analog_dat_F2[[1]]
        analog_dat_F3 <- analog_dat_F3[[1]]
        
        top_m_dt[1, c("top_1_fip", "top_2_fip", "top_3_fip") := as.list(analog_dat_F1$analog_NNs_county[1:3])]
        top_m_dt[2, c("top_1_fip", "top_2_fip", "top_3_fip") := as.list(analog_dat_F2$analog_NNs_county[1:3])]
        top_m_dt[3, c("top_1_fip", "top_2_fip", "top_3_fip") := as.list(analog_dat_F3$analog_NNs_county[1:3])]
        top_3_dt <- rbind(top_3_dt, top_m_dt)
      }
    }
    master_path <- paste0(data_dir, "top_3/")
    if (dir.exists(master_path) == F) { dir.create(path = master_path, recursive = T)}
    write.table(top_3_dt, 
                file = paste0(master_path, name_extension,  "_top_3.csv"), 
                row.names=FALSE, na="", col.names=TRUE, sep=",")
  }
}

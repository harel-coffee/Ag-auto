######################################################################
rm(list=ls())

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

f_loc_fips_st_cnty <- get_286_locs(f_loc_fips_st_cnty, h_loc_fips_st_cnty)

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

h_loc_fips_st_cnty <- h_loc_fips_st_cnty %>% 
                      filter(fips %in% hist_grid_count$fips) %>% 
                      data.table()

######################################################
# 
# To extract names of counties associated with each unique fips
# 

Min_fips_st_county <- data.table(read.csv(paste0(param_dir, "Min_fips_st_county_location.csv"), header=T, sep=",", as.is=T)) 
Min_fips_st_county <- within(Min_fips_st_county, remove(location))
Min_fips_st_county <- unique(Min_fips_st_county)

######################################################

######################################################

model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")
time_p <- c("2026-2050", "2051-2075", "2076-2095")
emissions <- c("rcp45", "rcp85")
precip_inclusions <- c("w_precip") #, "no_precip"
sigmas <- c("1_sigma", "2_sigma")
######################################################################
####
####         Set up directories
####
######################################################################

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/"
data_sub_dirs <- c("w_precip_rcp45", "w_precip_rcp85")

sub_dir <- data_sub_dirs[1]

###### features dir
feat_dir <- paste0(main_in, "00_databases/")

######################################################################
####
####         Read Features
####
######################################################################
CDD_precip_rcp45 <- data.table(readRDS(paste0(feat_dir, "CDD_precip_rcp45.rds")))
CDD_precip_rcp85 <- data.table(readRDS(paste0(feat_dir, "CDD_precip_rcp85.rds")))
hist_CDD_precip <- data.table(readRDS(paste0(feat_dir, "hist_CDD_precip.rds")))

add_time_periods_model <- function(dt){
  time_periods <- c("1950-2005", "2006-2025", "2026-2050", "2051-2075", "2076-2095")
  dt$time_period <- 0L
  dt$time_period[dt$year <= 2005] <- time_periods[1]
  dt$time_period[dt$year >= 2006 & dt$year <= 2025] <- time_periods[2]
  dt$time_period[dt$year >= 2026 & dt$year <= 2050] <- time_periods[3]
  dt$time_period[dt$year >= 2051 & dt$year <= 2075] <- time_periods[4]
  dt$time_period[dt$year >= 2076] <- time_periods[5]
  return(dt)
}

CDD_precip_rcp45 <- add_time_periods_model(CDD_precip_rcp45)
CDD_precip_rcp85 <- add_time_periods_model(CDD_precip_rcp85)


#######
####### Add fips to the features
#######
CDD_precip_rcp45 <- merge(CDD_precip_rcp45, f_loc_fips_st_cnty, by="location", all.x=TRUE)
CDD_precip_rcp85 <- merge(CDD_precip_rcp85, f_loc_fips_st_cnty, by="location", all.x=TRUE)
hist_CDD_precip <- merge(hist_CDD_precip, h_loc_fips_st_cnty, by="location", all.x=TRUE)

hist_CDD_precip$time_period <- "1979-2015"

# some counties have had less than min_grid grids in them, those
# are not in h_loc_fips_st_cnty, hence NA is produced, drop them:
hist_CDD_precip <- na.omit(hist_CDD_precip)

for (precip_stat in precip_inclusions){
  for (emission in emissions){
    for (sigma in sigmas){
      pe <- paste0(precip_stat, "_", emission)
      pe_path <- paste0(main_in, sigma, "/", pe, "/top_3/")
      t3_name <- paste0(pe_path, pe, "_top_3.csv")
      top_3 <- data.table(read.csv(t3_name, as.is=T, header=T, sep=","))
      for (f_fip in local_fips){
        if (emission == "rcp45"){ curr_dt <- CDD_precip_rcp45 } else { curr_dt <- CDD_precip_rcp85}
        curr_top_3 <- top_3 %>% filter(future_fip == f_fip)
        curr_top_3_f1 <- curr_top_3 %>% filter(time_period == "F1") %>% data.table()
        curr_top_3_f2 <- curr_top_3 %>% filter(time_period == "F2") %>% data.table()
        curr_top_3_f3 <- curr_top_3 %>% filter(time_period == "F3") %>% data.table()

        curr_fip_data <- curr_dt %>% filter(fips == f_fip)
        curr_future_fip_data_f1 <- curr_fip_data %>% filter(time_period == time_p[1]) %>% data.table()
        curr_future_fip_data_f2 <- curr_fip_data %>% filter(time_period == time_p[2]) %>% data.table()
        curr_future_fip_data_f3 <- curr_fip_data %>% filter(time_period == time_p[3]) %>% data.table()

        pp_f1 <- plot_f_h_2_features_all_models(curr_future_fip_data_f1, curr_top_3_f1, hist_CDD_precip)
        pp_f2 <- plot_f_h_2_features_all_models(curr_future_fip_data_f2, curr_top_3_f2, hist_CDD_precip)
        pp_f3 <- plot_f_h_2_features_all_models(curr_future_fip_data_f3, curr_top_3_f3, hist_CDD_precip)

        assign(x = "plot", 
               value={ggarrange(plotlist = list(pp_f1, pp_f2, pp_f3),
                                ncol = 3, nrow = 1,
                                common.legend = TRUE, 
                                legend = "bottom")})

        ggsave(filename = paste0(unique(curr_fip_data$st_county), ".png"), 
               plot = plot, 
               path = pe_path, device="png",
               dpi=300, width=40, height=100, unit="in", limitsize = FALSE)
      }
    }
  }
}











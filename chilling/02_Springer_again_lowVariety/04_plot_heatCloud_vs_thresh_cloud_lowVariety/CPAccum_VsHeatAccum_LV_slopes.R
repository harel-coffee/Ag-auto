rm(list=ls())

library(data.table)
library(dplyr)
library(ggpubr)
library(ggplot2)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
bloom_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
chill_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

bloom_core_source <- paste0(bloom_source_dir, "bloom_core.R")
bloom_plot_core_source <- paste0(bloom_source_dir, "bloom_plot_core.R")

chill_core_source <- paste0(chill_source_dir, "chill_core.R")

source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)

#############################################################
in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(bloom_source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/01_research_data/Ag_Papers_data/Chill_Paper/01_Springer_1/"

daily_CP_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/"
daily_vertDD_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/lowVariety/"

#############################################################
###
###               Read data off the disk
###
#############################################################

########
######## read parameters
########

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                              header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), as.is=TRUE)

########
######## read data
########

daily_CP     <- read.csv(paste0(daily_CP_dir, "daily_CP_for_CDF.csv"), as.is=TRUE)
daily_vertDD <- readRDS(paste0(daily_vertDD_dir, "triggerBased_vertDD_lowVariety.rds"))

heat_slopes <-  readRDS(paste0(daily_vertDD_dir, "slopes_maxDD_200_LV.rds"))
heat_slopes[, (c("coef")) := round(.SD, 1), .SDcols=c("coef")]
heat_slopes$str_slope <- paste0("m = ", heat_slopes$coef)

#____________________________________________________________
#
daily_CP <- daily_CP %>% 
            filter(emission != "modeled historical") %>% 
            data.table()

daily_CP[daily_CP$emission == "observed", "emission"] <- "Observed"

#____________________________________________________________
#
# add time periods to the data
#
daily_vertDD[daily_vertDD$time_period == "1979-2016", "time_period"] <- "Observed"

daily_CP_obs <- daily_CP %>%
                filter(emission == "Observed")

daily_CP_obs <- daily_CP %>%
                filter(emission == "Observed") %>%
                data.table()

daily_CP_F <- daily_CP %>%
              filter(emission != "Observed") %>%
              data.table()

daily_CP_obs$time_period <- "Observed"
daily_CP_F <- add_time_periods_model(daily_CP_F)

daily_CP <- rbind(daily_CP_obs, daily_CP_F)

#
# pick up observed and 2026-2099 time period
#

daily_vertDD <- pick_obs_and_F(daily_vertDD)
daily_CP <- pick_obs_and_F(daily_CP)

#############################################################
#
#              clean up and prepare each data table
#
#############################################################

daily_vertDD <- dplyr::left_join(x = daily_vertDD, 
                                 y = limited_locations, 
                                 by = "location")

x <- sapply(daily_CP$location, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
lat = x[1, ]
long = x[2, ]

daily_CP$lat <- lat
daily_CP$long <- long

daily_CP <- add_chill_sept_DoY(daily_CP)

daily_vertDD <- within(daily_vertDD, remove("location"))
daily_CP <- within(daily_CP, remove("location", "lat", "long"))

#############################################################

emissions <- c("RCP 4.5", "RCP 8.5")
setnames(daily_vertDD, old=c("city"), new=c("location"))
setnames(daily_CP, old=c("city"), new=c("location"))

################################################

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

daily_vertDD$location <- factor(daily_vertDD$location, levels = ict, order=TRUE)
daily_CP$location <-     factor(daily_CP$location,     levels = ict, order=TRUE)


daily_CP$emission <- factor(daily_CP$emission, 
                            levels = c("Observed", "RCP 4.5", "RCP 8.5"), 
                            order = TRUE)

daily_vertDD$emission <- factor(daily_vertDD$emission, 
                                levels = c("Observed", "RCP 4.5", "RCP 8.5"), 
                                order = TRUE)


daily_vertDD[daily_vertDD$time_period == "Observed", "time_period"] <- "Historical"
daily_CP[daily_CP$time_period == "Observed", "time_period"] <- "Historical"


tpo <- c("Historical", "2026-2050", "2051-2075", "2076-2099")
daily_CP$time_period <- factor(daily_CP$time_period, 
                               levels = tpo,
                               order = TRUE)

daily_vertDD$time_period <- factor(daily_vertDD$time_period, 
                                   levels = tpo, 
                                   order = TRUE)

####################################################################
start_time <- Sys.time()

daily_vertDD_obs <- daily_vertDD %>% filter(time_period == "Historical") %>% data.table()
daily_CP_obs <- daily_CP %>% filter(time_period == "Historical") %>% data.table()

daily_vertDD_obs45 <- daily_vertDD_obs
daily_vertDD_obs85 <- daily_vertDD_obs

daily_CP_obs45 <- daily_CP_obs
daily_CP_obs85 <- daily_CP_obs

daily_vertDD_obs45$emission <- "RCP 4.5"
daily_vertDD_obs85$emission <- "RCP 8.5"

daily_CP_obs45$emission <- "RCP 4.5"
daily_CP_obs85$emission <- "RCP 8.5"


daily_vertDD_RCP85 <- daily_vertDD %>% filter(emission == "RCP 8.5") %>% data.table()
daily_vertDD_RCP45 <- daily_vertDD %>% filter(emission == "RCP 4.5") %>% data.table()

daily_CP_RCP85 <- daily_CP %>% filter(emission == "RCP 8.5") %>% data.table()
daily_CP_RCP45 <- daily_CP %>% filter(emission == "RCP 4.5") %>% data.table()

daily_vertDD_RCP85 <- rbind(daily_vertDD_RCP85, daily_vertDD_obs85)
daily_vertDD_RCP45 <- rbind(daily_vertDD_RCP45, daily_vertDD_obs45)

daily_CP_RCP85 <- rbind(daily_CP_RCP85, daily_CP_obs85)
daily_CP_RCP45 <- rbind(daily_CP_RCP45, daily_CP_obs45)


rm(daily_CP, daily_vertDD)

####################################################################################################
#
# This for-loop takes care of RCP 4.5, and next one RCP 8.5.
# 

curr_daily_CP <- daily_CP_RCP45
curr_vertDD <- daily_vertDD_RCP45
curr_slopes <- heat_slopes %>%
              filter(emission == "RCP 4.5") %>%
              data.table()

##
##  pick up needed column so that melts work correctly.
##
##
needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")

curr_daily_CP <- subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
curr_vertDD   <- subset(curr_vertDD,   select = c(needed_cols, "vert_Cum_dd"))

############################################################

curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
curr_vertDD_melt <-  melt(data.table(curr_vertDD) ,  id = needed_cols)

merged_dt <- rbind(curr_dailyCP_melt, curr_vertDD_melt)


source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
merged_plt <- double_cloud_2_rows_Accum_VertDD_CP_Slopes_100Try(d1=merged_dt, 
                                                                slopes_dt=curr_slopes, 
                                                                full_CP_intcpt=50, 
                                                                heat_intcpt=25)

plot_dir <- paste0(plot_base_dir, "/CPAccum_heatAccum/")

if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }

ggsave(plot = merged_plt,
       filename = paste0("00_Acum_CPHeat_RCP45_lowVariety_Slopes.png"), 
       width = 11, height=10, units = "in", 
       dpi = 400, device = "png",
       path = plot_dir)
print (plot_dir)

#
# This for-loop takes care of RCP 8.5. and previous one RCP 4.5
#
curr_daily_CP <- daily_CP_RCP85
curr_vertDD <- daily_vertDD_RCP85
curr_slopes <- heat_slopes %>%
               filter(emission == "RCP 8.5") %>%
               data.table()
##
##  pick up needed column so that melts work correctly.
##
needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")

curr_daily_CP <-subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
curr_vertDD <-subset(curr_vertDD, select = c(needed_cols, "vert_Cum_dd"))

#################

curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
curr_vertDD_melt <-  melt(data.table(curr_vertDD) ,  id = needed_cols)

merged_dt <- rbind(curr_dailyCP_melt, curr_vertDD_melt)


source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
merged_plt <- double_cloud_2_rows_Accum_VertDD_CP_Slopes_100Try(d1=merged_dt, 
                                                                slopes_dt=curr_slopes, 
                                                                full_CP_intcpt=50, 
                                                                heat_intcpt=25)

plot_dir <- paste0(plot_base_dir, "/CPAccum_heatAccum/")

if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }

ggsave(plot = merged_plt,
       filename = paste0("00_Acum_CPHeat_RCP85_lowVariety_Slopes.png"), 
       width = 11, height=10, units = "in", 
       dpi = 400, device = "png",
       path = plot_dir)
print (plot_dir)

print(Sys.time() - start_time)



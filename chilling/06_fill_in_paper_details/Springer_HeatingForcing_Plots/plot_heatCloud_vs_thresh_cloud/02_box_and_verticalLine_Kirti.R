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

start_time <- Sys.time()

bloom_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
chill_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(bloom_source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/01_research_data/Ag_Papers/Chill_Paper/01_Springer_1/"

daily_CP_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/"
daily_vertDD_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/"
#############################################################
###
### 
###
#############################################################

bloom_core_source <- paste0(bloom_source_dir, "bloom_core.R")
bloom_plot_core_source <- paste0(bloom_source_dir, "bloom_plot_core.R")

chill_core_source <- paste0(chill_source_dir, "chill_core.R")

source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
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
######## daily CP data
########

daily_CP <- read.csv(paste0(daily_CP_dir, "daily_CP_for_CDF.csv"), as.is=TRUE)
daily_vertDD <- readRDS(paste0(daily_vertDD_dir, "triggerBased_vertDD.rds"))

# daily_vertDD_2 <- readRDS(paste0(daily_vertDD_dir, "triggerBased_vertDD_CrippsBloom_no50cut.rds"))

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

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

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

# apple types are for bloom, not thresholds
apple_types <- c("Cripps Pink")  # , "Gala", "Red Deli" 

# apple, cherry, pear; cherry 14 days shift, pear 7 days shift
fruit_types <- c("apple", "cherry", "pear") 
fruit_type <- "apple"

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


app_tp <- apple_types[1]

####################################################################################################

y_interceppt <- 73.3
cpRange_delta <- 0.2
min_cp_forRange <- y_interceppt - cpRange_delta
max_cp_forRange <- y_interceppt + cpRange_delta

min_DD_forRange <- 54
max_DD_forRange <- 56

####################################################################################################
#
# This for-loop takes care of RCP 4.5, and next one RCP 8.5.
# 
curr_daily_CP <- daily_CP_RCP45
curr_vertDD <- daily_vertDD_RCP45

##
##  pick up needed column so that melts work correctly.
##
needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")

curr_daily_CP <- subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
curr_vertDD   <- subset(curr_vertDD,   select = c(needed_cols, "vert_Cum_dd"))

############################################################
#
# Compute the stats for the little horizontal lines 
#

curr_daily_CP_narr <- curr_daily_CP %>% 
                      filter(cume_portions >= min_cp_forRange)

curr_daily_CP_narr <- curr_daily_CP_narr %>% 
                      filter(cume_portions <= max_cp_forRange)

CP_quan_25_75 <- curr_daily_CP_narr %>% 
                 group_by(location, emission, time_period ) %>% 
                 summarise(CP_quan_25 = quantile(chill_dayofyear, probs = 0.25),
                           CP_quan_75 = quantile(chill_dayofyear, probs = 0.75)) %>% 
                 data.table()

CP_quan_25_75$y_intercept <- y_interceppt
CP_quan_25_75$variable <- "cume_portions"

curr_vertDD_narr <- curr_vertDD %>% 
                    filter(vert_Cum_dd >= min_DD_forRange)

curr_vertDD_narr <- curr_vertDD_narr %>% 
                    filter(vert_Cum_dd <= max_DD_forRange)

vertDD_quan_25_100 <- curr_vertDD_narr %>% 
                      group_by(location, emission, time_period ) %>% 
                      summarise(vertDD_quan_25 = quantile(chill_dayofyear, probs = 0.25),
                                vertDD_quan_1 = quantile(chill_dayofyear, probs = 1)) %>% 
                       data.table()

vertDD_quan_25_100$vertDD_quan_1 <- vertDD_quan_25_100$vertDD_quan_25 + 1

vertDD_quan_25_100$y_intercept = 55
vertDD_quan_25_100$variable <- "vert_Cum_dd"

############################################################

curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
curr_vertDD_melt <-  melt(data.table(curr_vertDD) ,  id = needed_cols)

merged_dt <- rbind(curr_dailyCP_melt, curr_vertDD_melt)


source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
merged_plt <- box_and_LineKirti(CP_quans = CP_quan_25_75, 
                                vertDD_quants = vertDD_quan_25_100)

plot_dir <- paste0(plot_base_dir, "/CPAccum_heatAccum/")

if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }

ggsave(plot = merged_plt,
       filename = paste0("Acum_CPHeat_", gsub(" ", "_", app_tp), "_RCP45_Kirti.png"), 
       width = 13, height = 5, units = "in", 
       dpi = 400, device = "png",
       path = plot_dir)
print (plot_dir)

####################################################################
#
# This for-loop takes care of RCP 8.5. and previous one RCP 4.5
# 

curr_daily_CP <- daily_CP_RCP85
curr_vertDD <- daily_vertDD_RCP85

##
##  pick up needed column so that melts work correctly.
##
needed_cols <- c("location", "emission", "model", "time_period", "chill_dayofyear")

curr_daily_CP <-subset(curr_daily_CP, select = c(needed_cols, "cume_portions"))
curr_vertDD <-subset(curr_vertDD, select = c(needed_cols, "vert_Cum_dd"))
############################################################
#
# Compute the stats for the little horizontal lines 
#
curr_daily_CP_narr <- curr_daily_CP %>% 
                      filter(cume_portions >= min_cp_forRange)

curr_daily_CP_narr <- curr_daily_CP_narr %>% 
                      filter(cume_portions <= max_cp_forRange)

CP_quan_25_75 <- curr_daily_CP_narr %>% 
                 group_by(location, emission, time_period ) %>% 
                 summarise(CP_quan_25 = quantile(chill_dayofyear, probs = 0.25),
                           CP_quan_75 = quantile(chill_dayofyear, probs = 0.75)) %>% 
                 data.table()

CP_quan_25_75$y_intercept <- y_interceppt
CP_quan_25_75$variable <- "cume_portions"

curr_vertDD_narr <- curr_vertDD %>% 
                    filter(vert_Cum_dd >= min_DD_forRange)

curr_vertDD_narr <- curr_vertDD_narr %>% 
                    filter(vert_Cum_dd <= max_DD_forRange)

vertDD_quan_25_100 <- curr_vertDD_narr %>% 
                      group_by(location, emission, time_period ) %>% 
                      summarise(vertDD_quan_25 = quantile(chill_dayofyear, probs = 0.25),
                                vertDD_quan_1 = quantile(chill_dayofyear, probs = 1)) %>% 
                      data.table()

vertDD_quan_25_100$vertDD_quan_1 <- vertDD_quan_25_100$vertDD_quan_25 + 1
vertDD_quan_25_100$y_intercept = 55
vertDD_quan_25_100$variable <- "vert_Cum_dd"

############################################################

curr_dailyCP_melt <- melt(data.table(curr_daily_CP), id = needed_cols)
curr_vertDD_melt <-  melt(data.table(curr_vertDD) ,  id = needed_cols)

merged_dt <- rbind(curr_dailyCP_melt, curr_vertDD_melt)


source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)
merged_plt <- box_and_LineKirti(CP_quans = CP_quan_25_75, 
                                vertDD_quants = vertDD_quan_25_100)

plot_dir <- paste0(plot_base_dir, "/CPAccum_heatAccum/")

if (dir.exists(plot_dir) == F) {
    dir.create(path = plot_dir, recursive = T)
    print (plot_dir)
  }

ggsave(plot = merged_plt,
       filename = paste0("Acum_CPHeat_", gsub(" ", "_", app_tp), "_RCP85_Kirti.png"), 
       width = 13, height=5, units = "in", 
       dpi = 400, device = "png",
       path = plot_dir)
print (plot_dir)


print(Sys.time() - start_time)



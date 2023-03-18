###
###   March 2, 2021. New plots (first attempt) for Springer.
###   This is a copy of d_facet_2rows_cloudy_thresh_vs_75PercentCP_HeatTriggerLine.R
###   Here we use 50 accumulated GDD as the trigger of heat accumulation as opposed 
###   75% of accumulated CP.
###
###
#####################################
rm(list=ls())

library(data.table)
library(dplyr)
library(purrr) # detect_index() is in this library
library(ggpubr)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
bloom_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
chill_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

source_1 <- paste0(bloom_source_dir, "bloom_core.R")
source_2 <- paste0(bloom_source_dir, "bloom_plot_core.R")
source_3 <- paste0(chill_source_dir, "chill_core.R")

source(source_1)
source(source_2)
source(source_3)

#############################################################
###
###
###
#############################################################

in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(bloom_source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/01_research_data/Ag_Papers/Chill_Paper/01_Springer_1/"


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
######## threshold data
########

thresh <- readRDS(paste0(in_dir, "sept_summary_comp.rds"))

########
######## Heat Trigger data
########
head_dir <- paste0(in_dir, "01_binary_to_bloom/observed/")

file_names <- c("bloom_48.40625_-119.53125.rds", "bloom_44.03125_-123.09375.rds",
                "bloom_46.03125_-118.34375.rds", "bloom_46.59375_-120.53125.rds")


heatTrig_obs = data.table()
for (file in file_names){
  A <- readRDS(paste0(head_dir, file))
  heatTrig_obs <- rbind(heatTrig_obs, A)
}


heatTrig_obs <- add_location(heatTrig_obs)
thresh <- add_location(thresh)


heatTrig_obs <- within(heatTrig_obs, remove(lat, long))
suppressWarnings({thresh <- within(thresh, remove(lat, long))})

###########################
###########################   Find the DoY at which 50 GDD is accumulated
###########################

#
#  The DoY in the following is from calendar year.
#  also the cut_off is aplied to vert_Cum_dd in C. So, if it needs to change
#  to F. go in ther and do it.
#

source(source_3)
heatTrig_obs <- Accum_GDD_DoYDetector_basedOnCaneldarYear(data = heatTrig_obs, 
                                                          data_type = "observed",
                                                          cut_off = 55)


#### covert DoY to chillDoY
#
# 1. add month to the table
#
heatTrig_obs <- heatTrig_obs %>%
                mutate(month = case_when(heatTriggerThresh %in% c(1:31) ~ 1,
                                         heatTriggerThresh %in% c(32:59) ~ 2,
                                         heatTriggerThresh %in% c(60:90) ~ 3,
                                         heatTriggerThresh %in% c(91:120) ~ 4,
                                         heatTriggerThresh %in% c(121:151) ~ 5,
                                         heatTriggerThresh %in% c(152:181) ~ 6,
                                         heatTriggerThresh %in% c(182:212) ~ 7,
                                         heatTriggerThresh %in% c(213:243) ~ 8,
                                         heatTriggerThresh %in% c(244:273) ~ 9,
                                         heatTriggerThresh %in% c(274:304) ~ 10,
                                         heatTriggerThresh %in% c(305:334) ~ 11,
                                         heatTriggerThresh %in% c(335:366) ~ 12)
                        )
#
#  Convert DoY to chillDoy
#
heatTrig_obs$chill_doy <- 1
for (m in (1:8)){
  heatTrig_obs$chill_doy[heatTrig_obs$month==m] <- heatTrig_obs$heatTriggerThresh[heatTrig_obs$month==m] + 122
}

for (m in (9:12)){
  heatTrig_obs$chill_doy[heatTrig_obs$month==m] <- heatTrig_obs$heatTriggerThresh[heatTrig_obs$month==m] - 244
}


####
#### Take median across years
####
heatTrig_obs <- heatTrig_obs %>%
                group_by(location) %>%
                summarise(median_heatTriggerThresh = median(chill_doy))%>%
                data.table()

###########################
###########################  Filter needed locations
###########################

needed_cities <- c("Omak", "Yakima", "Walla Walla", "Eugene")
limited_locations <- limited_locations %>% 
                     filter(city %in% needed_cities) %>% 
                     data.table()

thresh <- thresh %>% 
          filter(location %in% limited_locations$location) %>% 
          data.table()

heatTrig_obs <- heatTrig_obs %>% 
                filter(location %in% limited_locations$location) %>% 
                data.table()

###########################
###########################  add cities to the data
###########################

#
#  It seems merging the file as follows eliminates the need for
#  the step above (filtering the location will be taken care of by merge(x, y, by="location"))
#
thresh <-       dplyr::left_join(x = thresh,       y = limited_locations, by = "location")
heatTrig_obs <- dplyr::left_join(x = heatTrig_obs, y = limited_locations, by = "location")

thresh <- data.table(thresh)
heatTrig_obs <- data.table(heatTrig_obs)

##
## pick up observed and 2026-2099 time period, i.e. toss modeled history
##
thresh <- pick_obs_and_F(thresh)

#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")

# apple types are for bloom, not thresholds
apple_types <- c("Cripps Pink")  # , "Gala", "Red Deli" 

# apple, cherry, pear; cherry 14 days shift, pear 7 days shift
fruit_types <- c("apple", "cherry", "pear") 
fruit_type <- "apple"
remove_NA <- "no"

thresh <- within(thresh, remove(location))
heatTrig_obs <- within(heatTrig_obs, remove(location))

setnames(thresh, old=c("city"), new=c("location"))
setnames(heatTrig_obs, old=c("city"), new=c("location"))

###############################################################
# em <- emissions[2]
# app_tp <- apple_types[1]
# thresh_cut <- 75
# loc <- unique(thresh$location)[1]

start_time <- Sys.time()
plot_threshols <- seq(20, 75, 5) # seq(25, 75, 5)
plot_threshols <- c(45, 75)      # seq(25, 75, 5)

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")


thresh$location <-       factor(thresh$location,       levels = ict, order=TRUE)
heatTrig_obs$location <- factor(heatTrig_obs$location, levels = ict, order=TRUE)

####################################################################

thresh_RCP85 <- thresh %>% filter(emission == "RCP 8.5")
thresh_RCP45 <- thresh %>% filter(emission == "RCP 4.5")

# thresh_RCP85$location  <- paste0(thresh_RCP85$location, " - ", thresh_RCP85$emission)
# thresh_RCP45$location  <- paste0(thresh_RCP45$location, " - ", thresh_RCP45$emission)

thresh_RCP85 <- within(thresh_RCP85, remove(emission))
thresh_RCP45 <- within(thresh_RCP45, remove(emission))


ict <- c("Omak - RCP 8.5", "Yakima - RCP 8.5", "Walla Walla - RCP 8.5", "Eugene - RCP 8.5")
ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
thresh_RCP85$location <- factor(thresh_RCP85$location, levels = ict, order=TRUE)


ict <- c("Omak - RCP 4.5", "Yakima - RCP 4.5", "Walla Walla - RCP 4.5", "Eugene - RCP 4.5")
ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
thresh_RCP45$location <- factor(thresh_RCP45$location, levels = ict, order=TRUE)

rm(thresh)

####################################################################################################
##
##  For testing
##
thresh_cut <- plot_threshols[1]
app_tp <- apple_types[1]
####################################################################################################
#
# This for-loop takes care of RCP 4.5, and next one RCP 8.5.
#
for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)
  curr_thresh <- thresh_RCP45

  curr_thresh <- subset(curr_thresh, 
                        select=c("location", "chill_season",
                                 "model",
                                 "time_period", col_name))

  #############################################################
  #
  #        REMOVE NAs
  #
  if (remove_NA == "yes"){
    curr_thresh <- curr_thresh %>% 
                   filter(get(col_name)<365) %>% 
                   data.table()
  }

  for (app_tp in apple_types){

    if (paste0("thresh_", thresh_cut) %in% colnames(curr_thresh)){
      setnames(curr_thresh, 
               old=c(paste0("thresh_", thresh_cut)), 
               new=c("thresh"))
    }

    curr_thresh_melt <- melt(data.table(curr_thresh), 
                             id = c("location", "chill_season", 
                                    "time_period", "model"))

    merged_dt <- rbind(curr_thresh_melt)
    merged_dt <- merged_dt %>% filter(time_period=="future")
    
    source(source_1)
    source(source_2)
    merged_plt <- ThreshCloud_2_rows_forForcing(d1 = merged_dt, trigger_dt=heatTrig_obs)

    if (remove_NA=="yes"){
      LP <- "NA_removed_"
      } else{
      LP <- "NA_NOTremoved_"
    }

    plot_dir <- paste0(plot_base_dir, "/thresh_forceLine/", LP, "fixed_y/")
    
    if (dir.exists(plot_dir) == F) {
        dir.create(path = plot_dir, recursive = T)
        print (plot_dir)
      }

    ggsave(plot = merged_plt,
           filename = paste0(fruit_type, "_RCP45_",
                             gsub(" ", "_", app_tp), "_", thresh_cut, "CP_2rows_VertDDTrigger.png"), 
           width=13, height=10, units = "in", 
           dpi = 400, device = "png",
           path=plot_dir)
    print (plot_dir)
  }
}

#
# This for-loop takes care of RCP 8.5, and previous one RCP 4.5.
#

for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)
  curr_thresh <- thresh_RCP85

  curr_thresh <- subset(curr_thresh, 
                        select=c("location", "chill_season",
                                 "model",
                                 "time_period", col_name))

  #############################################################
  #
  #        REMOVE NAs
  #
  if (remove_NA == "yes"){
    curr_thresh <- curr_thresh %>% 
                   filter(get(col_name)<365) %>% 
                   data.table()
  }

  for (app_tp in apple_types){

    if (paste0("thresh_", thresh_cut) %in% colnames(curr_thresh)){
      setnames(curr_thresh, 
               old=c(paste0("thresh_", thresh_cut)), 
               new=c("thresh"))
    }

    curr_thresh_melt <- melt(data.table(curr_thresh), 
                             id = c("location", "chill_season", 
                                    "time_period", "model"))

    merged_dt <- rbind(curr_thresh_melt)
    merged_dt <- merged_dt %>% filter(time_period=="future")

    source(source_1)
    source(source_2)
    merged_plt <- ThreshCloud_2_rows_forForcing(d1 = merged_dt, trigger_dt=heatTrig_obs)

    if (remove_NA=="yes"){
      LP <- "NA_removed_"
      } else{
      LP <- "NA_NOTremoved_"
    }

    plot_dir <- paste0(plot_base_dir, "/thresh_forceLine/", LP, "fixed_y/")
    
    if (dir.exists(plot_dir) == F) {
        dir.create(path = plot_dir, recursive = T)
        print (plot_dir)
      }

    ggsave(plot = merged_plt,
           filename = paste0(fruit_type, "_RCP85_",
                             gsub(" ", "_", app_tp), "_", thresh_cut, "CP_2rows_VertDDTrigger.png"), 
           width=13, height=10, units = "in", 
           dpi = 400, device = "png",
           path=plot_dir)
    print (plot_dir)
  }
}


print(Sys.time() - start_time)



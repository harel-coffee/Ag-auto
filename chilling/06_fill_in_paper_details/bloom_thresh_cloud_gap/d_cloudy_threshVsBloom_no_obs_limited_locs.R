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
source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(source_dir, "parameters/")
plot_base_dir <- "/Users/hn/Documents/01_research_data/bloom/plots/"
############################################################
###
###             Aeolus source
###
############################################################
source_dir <- "/home/hnoorazar/bloom_codes/"

###   Aeolus Directories
base <- "/data/hydro/users/Hossein/bloom/"
in_dir <- paste0(base, "03_merge_02_Step/")
param_dir <- paste0(source_dir, "parameters/")
plot_base_dir <- paste0(base, "04_bloom_vs_frost_plot/limited_locs/") # It was CM_locs

#############################################################
###
###              
###
#############################################################

source_1 <- paste0(source_dir, "bloom_core.R")
source_2 <- paste0(source_dir, "bloom_plot_core.R")
source(source_1)
source(source_2)
#############################################################
###
###               Read data off the disk
###
#############################################################
#
# for sake of iteration
#
# close_locs <- read.csv(file = paste0(param_dir, "close_locs_4_bloom.csv"), 
#                         header=TRUE, as.is=TRUE)
# close_locs <- within(close_locs, remove(lat, long, latDiff, distance, longDiff))

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                        header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

# limited_locations <- limited_locations %>% 
#                      filter(city %in% c("Omak", "Richland"))


# close_locs <- close_locs %>% 
#                filter(city %in% c("Corvallis", "Eugene", "Hillsboro" , "Hood River", "Salem",
#                                    "Walla Walla", "Wenatchee", "Yakima"))

# close_locs <- rbind(close_locs, limited_locations)
#
# convert chill day of year to readable day of year!
#
chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), 
                          as.is=TRUE)
#############################################################
bloom <- readRDS(paste0(in_dir, "fullbloom_50percent_day.rds"))
thresh <- readRDS(paste0(in_dir, "sept_summary_comp.rds"))

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

bloom <- pick_obs_and_F(bloom)
thresh <- pick_obs_and_F(thresh)

bloom <- bloom %>% filter(time_period == "future") %>% data.table()
thresh <- thresh %>% filter(time_period == "future") %>% data.table()

#############################################################
#
#              clean up each data table
#
#############################################################
bloom <- within(bloom, remove(year, month, day, dayofyear, bloom_perc))

# for sake of iteration:
bloom <- add_location(bloom)
thresh <- add_location(thresh)

suppressWarnings({bloom <- within(bloom, remove(lat, long))})
suppressWarnings({thresh <- within(thresh, remove(lat, long))})

bloom <- bloom %>% filter(location %in% limited_locations$location)
thresh <- thresh %>% filter(location %in% limited_locations$location)

bloom <- dplyr::left_join(x = bloom, y = limited_locations, by = "location")
thresh <- dplyr::left_join(x = thresh, y = limited_locations, by = "location")

bloom <- data.table(bloom)
thresh <- data.table(thresh)

#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")
apple_types <- c("Cripps Pink", "Gala", "Red Deli") # , "Gala", "Red Deli"

# apple, cherry, pear; cherry 14 days shift, pear 7 days shift
fruit_types <- c("apple", "cherry", "pear") 
fruit_type <- "cherry"
remove_NA <- "no"

# shift the bloom days
if (fruit_type == "cherry"){
   bloom$chill_doy <- bloom$chill_doy-14
   bloom <- bloom %>% filter(chill_doy>=0)
   # This is done just for purpose of for loop
   # apple_types <- c("Cripps Pink") 
   } else if (fruit_type == "pear"){
    bloom$chill_doy <- bloom$chill_doy-7
    bloom <- bloom %>% filter(chill_doy>=0)
    # This is done just for purpose of for loop
    # apple_types <- c("Cripps Pink") 
}

setnames(bloom, old=c("chill_doy"), new=c("fifty_perc_chill_DoY"))
bloom <- data.table(bloom)


bloom <- within(bloom, remove(location))
thresh <- within(thresh, remove(location))

setnames(bloom, old=c("city"), new=c("location"))
setnames(thresh, old=c("city"), new=c("location"))
###############################################################


start_time <- Sys.time()
plot_threshols <- seq(20, 75, 5) # seq(25, 75, 5)
plot_threshols <- c(45, 75)      # seq(25, 75, 5)

##########################################################################################
# 
# For testing
#

thresh_cut = plot_threshols[1]
loc = unique(bloom$location)[1]
em = emissions[1]
app_tp = apple_types[1]


##########################################################################################

for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)

  for (loc in unique(bloom$location)){
    for (em in emissions){
      curr_thresh <- thresh %>% 
                     filter(location==loc & emission==em) %>% 
                     data.table()
      #############################################################
      ##
      ##             Thresh plot here
      ##
      curr_thresh <- subset(curr_thresh, 
                            select=c("location", "chill_season",
                                     "model", "emission",
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
      #############################################################
      ##
      ##             END of Thresh plot
      ##
      #############################################################
      #############################################################
      #
      #            Bloom plots
      #

      for (app_tp in apple_types){
        curr_bloom <- bloom %>% 
                      filter(location==loc & emission==em & 
                             fruit_type == gsub("\ ", "_", 
                                               tolower(app_tp))) %>% 
                      data.table()

        if (fruit_type=="apple"){
             title_ <- paste0(app_tp, " bloom shift (")
           } else {
             title_ <- paste0(fruit_type, " bloom shift (")
        }
        #######*********#######*********#######*********#######*********
        ######################################################
        #                                                     #
        # layover -- layover -- layover -- layover -- layover #
        #          organize for layover plot                  #
        # layover -- layover -- layover -- layover -- layover #
        #                                                     #
        #######################################################

        suppressWarnings({curr_bloom<-within(curr_bloom, 
                                             remove(fruit_type))})
        setcolorder(curr_bloom, c("location", 
                                  "time_period", "chill_season", 
                                  "model", "emission", 
                                  "fifty_perc_chill_DoY"))

        if (paste0("thresh_", thresh_cut) %in% colnames(curr_thresh)){
          setnames(curr_thresh, 
                   old=c(paste0("thresh_", thresh_cut)), 
                   new=c("thresh"))
        }

        curr_thresh_melt <- melt(curr_thresh, 
                                 id=c("location", 
                                      "chill_season", "time_period", 
                                      "model", "emission"))

        curr_bloom_melt <- melt(curr_bloom, 
                                id=c("location", 
                                     "chill_season", "time_period", 
                                     "model", "emission"))

        merged_dt <- rbind(curr_thresh_melt, curr_bloom_melt)
        merged_dt <- merged_dt %>% filter(time_period=="future")

        if (fruit_type=="apple"){
           title_ <- paste0(thresh_cut, " CP threshold and ", app_tp, " bloom shifts")
           } else{
            title_ <- paste0(thresh_cut, " CP threshold and ", fruit_type, " bloom shifts")
        }

        merged_plt <- double_cloud(d1=merged_dt) + ggtitle(lab=paste0(title_)) 

        if (remove_NA=="yes"){
          LP <- "NA_removed"
          } else{
          LP <- "NA_NOTremoved"
        }

        bloom_thresh_plot_dir <- paste0(plot_base_dir, "limited_locations_", LP,
                                        "/bloom_thresh_in_one/no_obs/y_May_15/", 
                                         fruit_type, "/", col_name, "/")
        if (dir.exists(bloom_thresh_plot_dir) == F) {
            dir.create(path = bloom_thresh_plot_dir, recursive = T)
            print (bloom_thresh_plot_dir)
          }

        ggsave(plot = merged_plt,
               filename = paste0(gsub(" ", "_", loc), "_", 
                                 gsub(" ", "_", gsub("\\.", "", em)), "_", 
                                 gsub(" ", "_", app_tp), "_", thresh_cut, "CP.png"), 
               width = 10, height=6, units = "in", 
               dpi = 600, device = "png",
               path = bloom_thresh_plot_dir)
        print (bloom_thresh_plot_dir)
      }
    }
  }
}

print(Sys.time() - start_time)



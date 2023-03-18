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
plot_base_dir <- paste0(base, "04_bloom_vs_frost_plot/")
#############################################################
###
###               Read data off the disk
###
#############################################################
#
# for sake of iteration
#
location_ls <- read.table(file = paste0(param_dir, 
                                        "file_list.txt"), 
                          header=FALSE, as.is=TRUE)
location_ls <- as.vector(location_ls$V1)
location_ls <- gsub("data_", "", location_ls)

#
# convert chill day of year to readable day of year!
#
chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), 
                          as.is=TRUE)
#############################################################

frost <- readRDS(paste0(in_dir, "first_frost.rds"))
bloom <- readRDS(paste0(in_dir, "fullbloom_50percent_day.rds"))
thresh <- readRDS(paste0(in_dir, "sept_summary_comp.rds"))

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
bloom <- pick_obs_and_F(bloom)
thresh <- pick_obs_and_F(thresh)

#############################################################
#
#              clean up each data table
#
#############################################################
frost <- within(frost, remove(tmin))
bloom <- within(bloom, remove(year, month, day, dayofyear, 
                              bloom_perc))

# for sake of iteration:
frost <- add_location(frost)
bloom <- add_location(bloom)
thresh <- add_location(thresh)

suppressWarnings({frost <- within(frost, remove(lat, long))})
suppressWarnings({bloom <- within(bloom, remove(lat, long))})
suppressWarnings({thresh <- within(thresh, remove(lat, long))})

print(length(unique(thresh$chill_season)))
print(length(unique(bloom$chill_season)))
print(length(unique(frost$chill_season)))

print(length(unique(thresh$location)))
print(length(unique(bloom$location)))
print(length(unique(frost$location)))

print(length(unique(thresh$model)))
print(length(unique(bloom$model)))
print(length(unique(frost$model)))
#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")
apple_types <- c("Cripps Pink", "Gala", "Red Deli")

# apple, Cherry, Pear; Cherry 14 days shift, Pear 7 days shift
fruit_type <- "apple"
remove_NA <- "yes" 

# shift the bloom days
if (fruit_type == "Cherry"){
   bloom$dayofyear <- bloom$dayofyear-14
   bloom <- bloom %>% filter(dayofyear>=0)
   # This is done just for purpose of for loop
   # apple_types <- c("Cripps Pink") 
   } else if (fruit_type == "Pear"){
    bloom$dayofyear <- bloom$dayofyear-7
    bloom <- bloom %>% filter(dayofyear>=0)
    # This is done just for purpose of for loop
    # apple_types <- c("Cripps Pink") 
}

setnames(bloom, old=c("chill_doy"), new=c("fifty_perc_chill_DoY"))
bloom <- data.table(bloom)

###############################################################
em <- emissions[2]
app_tp <- apple_types[1]
thresh_cut <- 75
loc <- location_ls[1]

# frost <- frost %>% filter(location == loc)
# bloom <- bloom %>% filter(location == loc)
# thresh <- thresh %>% filter(location == loc)
# location_ls <- c(loc)

start_time <- Sys.time()
plot_threshols <- 75 # seq(50, 75, 5)

for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)
  frost_plot_dir <- paste0(plot_base_dir, 
                           "frost_plots/", "/")

  if (dir.exists(frost_plot_dir) == F) {
    dir.create(path = frost_plot_dir, recursive = T)
  }

  for (loc in location_ls){
    curr_lat <- substr(unique(loc), 1, 8)
    curr_long <- substr(unique(loc), 11, 19)
    for (em in emissions){
      curr_frost <- frost %>% 
                    filter(location==loc & emission==em) %>% 
                    data.table()

      curr_thresh <- thresh %>% 
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
      ##
      ##             End of Frost plot here
      ##
      #############################################################
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
      #######*********#######*********#######*********#######*********
      
      # thresh_plt <- cloudy_frost(d1=curr_thresh, 
      #                            colname=paste0("thresh_", thresh_cut), 
      #                            fil=paste0(thresh_cut, " CP ", 
      #                                       "threshold")) +
      #               ggtitle(lab=paste0(thresh_cut," CP shift (",
      #                                  curr_lat, " N, ",
      #                                  curr_long, " W), ",
      #                                  em))

      # just_thresh_plot_dir <- paste0(plot_base_dir, 
      #                                "just_thresh_plots/no_obs/", 
      #                                 fruit_type, "/thresh_",
      #                                 thresh_cut, "/")
      # if (dir.exists(just_thresh_plot_dir)==F){
      #     dir.create(path=just_thresh_plot_dir, recursive=T)}

      # ggsave(plot = thresh_plt,
      #        filename = paste0(loc, "_", gsub(" ", "_", em), ".png"), 
      #        width = 8, height=6, units = "in", 
      #        dpi=400, device = "png",
      #        path=just_thresh_plot_dir)
      #######*********#######*********#######*********#######*********
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
        #######*********#######*********#######*********#######*********
        # just_bloom <- cloudy_frost(d1=curr_bloom, 
        #                            colname="fifty_perc_chill_DoY", 
        #                            fil="bloom") +
        #               ggtitle(lab=paste0(title_, curr_lat, " N, ",
        #                                curr_long, " W), ",
        #                                em))

        # just_bloom_plot_dir <- paste0(plot_base_dir, 
        #                              "just_bloom_plots/no_obs/", 
        #                               fruit_type, "/")
        # if (dir.exists(just_bloom_plot_dir) == F) {
        #     dir.create(path = just_bloom_plot_dir, recursive = T)}
        
        # ggsave(plot=just_bloom,
        #        filename = paste0(loc, "_", 
        #                          gsub(" ", "_", em), "_", 
        #                          gsub(" ", "_", app_tp), ".png"), 
        #        width = 8, height=6, units = "in", 
        #        dpi=400, device = "png",
        #        path=just_bloom_plot_dir)
        #######*********#######*********#######*********#######*********
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
           title_ <- paste0(thresh_cut, " CP threshold ", 
                            " and bloom shifts ", "(", app_tp,", ")
           } else{
            title_ <- paste0(thresh_cut, " CP threshold and ", 
                             fruit_type, " bloom shifts")
        }

        merged_plt <- double_cloud(d1=merged_dt) + 
                      ggtitle(lab=paste0(title_, curr_lat, " N, ",
                                       curr_long, " W), ",
                                       em))

        bloom_thresh_plot_dir <- paste0(plot_base_dir, 
                                        "bloom_thresh_in_one/no_obs/", 
                                        fruit_type, "/", col_name, "/")
        if (dir.exists(bloom_thresh_plot_dir) == F) {
            dir.create(path = bloom_thresh_plot_dir, recursive = T)}
        ggsave(plot=merged_plt,
               filename = paste0(loc, "_", 
                                 gsub(" ", "_", em), "_", 
                                 gsub(" ", "_", app_tp), ".png"), 
               width=10, height=6, units = "in", 
               dpi=400, device = "png",
               path=bloom_thresh_plot_dir)
      }
    }
  }
}

print(Sys.time() - start_time)



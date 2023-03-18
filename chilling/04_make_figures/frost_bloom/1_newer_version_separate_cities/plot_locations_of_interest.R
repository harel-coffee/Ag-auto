rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_1)
source(source_path_2)

#######################################################################################
                         #                            #
                         #       Functions here       #
                         #                            #
                         ##############################

#######################################################################################
dues <- c("Dec", "Jan", "Feb")
due <- dues[1]
doY_dir <- "/Users/hn/Documents/GitHub/Ag/"
DoY_Map <- read.csv(paste0(doY_dir, "DoY_Map.csv"), as.is=TRUE)
dayCountStartSept1 <- read.csv(paste0(doY_dir, "dayCountStartSept1.csv"), as.is=TRUE)

needed_TP <- c("1979-2015", "2026-2050", "2051-2075", "2076-2099")

for (due in dues){
  #######################################################################################
  param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
  LOI <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T))

  #######################################################################################
  # Read Data
  
  data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
  data_dir <- paste0(data_dir, due, "/")

  first_frost <- data.table(readRDS(paste0(data_dir, "first_frost_till_", due, ".rds")))
  fifth_frost <- data.table(readRDS(paste0(data_dir, "fifth_frost_till_", due, ".rds")))

  first_frost <- first_frost %>% filter(year != 1949) %>% data.table()
  fifth_frost <- fifth_frost %>% filter(year != 1949) %>% data.table()

  first_frost <- first_frost %>% filter(time_period %in% needed_TP)
  fifth_frost <- fifth_frost %>% filter(time_period %in% needed_TP)
  
  out_dir <- paste0(data_dir, "cleaner/"); print(out_dir)
  if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
  }

  saveRDS(first_frost, paste0(out_dir, "first_frost_till_", due, ".rds"))
  saveRDS(fifth_frost, paste0(out_dir, "fifth_frost_till_", due, ".rds"))
  #######################################################################################
                           #                            #
                           #    box plot of all locs    #
                           #                            #
                           ##############################
  print(length(unique(first_frost$location)))
  print(length(unique(fifth_frost$location)))
  annot_text <- "All locations (2358) and 19 models are included here."
  orig_first_frost <- first_frost
  orig_fifth_frost <- fifth_frost
  
  em <- "RCP 8.5"
  for (em in c("RCP 4.5", "RCP 8.5")){
    first_frost <- orig_first_frost %>% filter(emission == em) %>% data.table()
    fifth_frost <- orig_fifth_frost %>% filter(emission == em) %>% data.table()

    first_frost_all_locs_box <- boxplot_frost_dayofyear(dt=first_frost, 
                                                        kth_day=1, 
                                                        sub_title=annot_text)
    
    fifth_frost_all_locs_box <- boxplot_frost_dayofyear(dt=fifth_frost, 
                                                        kth_day=5, 
                                                        sub_title=annot_text)

    ggsave(path = out_dir,
           plot = first_frost_all_locs_box, 
           filename = paste0(due, "_1st_frost_all_locs_", em ,".png"), 
           width = 5.5, height = 4, units = "in", 
           dpi = 400, device = "png")

    ggsave(path = out_dir, 
           plot = fifth_frost_all_locs_box, 
           filename = paste0(due, "_5th_frost_all_locs_", em ,".png"), 
           width = 5.5, height = 4, units = "in", 
           dpi = 400, device = "png")


    # assembeled <- ggarrange(plotlist = list(first_frost_all_locs_box, fifth_frost_all_locs_box), 
    #                         ncol = 1, nrow = 2, 
    #                         widths = c(1, 2.3) , heights = 1,
    #                         common.legend = TRUE, legend = "bottom")

    # ggsave(path = out_dir, 
    #        plot = assembeled, 
    #        filename = paste0(due, "_frost_all_locs.png"), 
    #        width = 6, height = 6, units = "in", 
    #        dpi = 400, device = "png")

    ########################################################
    #
    #          Filter locations of interest data
    #

    first_frost_limited_orig <- pick_single_cities_by_location(dt=first_frost, city_info=LOI)
    fifth_frost_limited_orig <- pick_single_cities_by_location(dt=fifth_frost, city_info=LOI)
    #########################################################################################
    #######################################################################################
                             #                                       #
                             #  Box plot of locations of interest    #
                             #                                       #
                             #########################################
    ct <- "Omak"
    for (ct in unique(fifth_frost_limited_orig$city)){
      first_frost_limited <- first_frost_limited_orig %>% filter(city == ct) %>% data.table()
      fifth_frost_limited <- fifth_frost_limited_orig %>% filter(city == ct) %>% data.table()
      
      first_frost_select_locs_box <- boxplot_frost_dayofyear(dt=first_frost_limited,
                                                             kth_day=1, 
                                                             sub_title=" ")

      fifth_frost_select_locs_box <- boxplot_frost_dayofyear(dt=fifth_frost_limited,
                                                             kth_day=5, 
                                                             sub_title=" ")

      ggsave(plot = first_frost_select_locs_box, 
             filename = paste0(due, "_1st_frost_", ct, "_", em, ".png"), 
             width = 5.2, height = 3.4, units = "in", 
             dpi = 400, device = "png",
             path = out_dir)

      ggsave(plot = fifth_frost_select_locs_box, 
             filename = paste0(due, "_5th_frost_", ct,  "_", em, ".png"), 
             width = 5.2, height = 3.4, units = "in", 
             dpi = 400, device = "png",
             path = out_dir)

      # assembeled_limited <- ggarrange(plotlist = list(first_frost_select_locs_box, 
      #                                                 fifth_frost_select_locs_box), 
      #                                                 ncol = 1, nrow = 2, 
      #                                                 widths = c(1, 2.3) , heights = 1,
      #                                                 common.legend = TRUE, legend = "bottom")
      # ggsave(plot = assembeled_limited, 
      #        filename = paste0(due, "_frost_selected_locs.png"), 
      #        width = 10, height = 8, units = "in", 
      #        dpi = 400, device = "png",
      #        path = data_dir)

    ###################################################################################
    #
    #         Take Median accross models so we have one model per location
    #
    ###################################################################################
    #
    # group by time period, location, emission and year and take medians
    # This way just models are killed per year, and time series can be plotted.
    #

    first_frost_limited_obs <- first_frost_limited %>% filter(model == "Observed")
    fifth_frost_limited_obs <- fifth_frost_limited %>% filter(model == "Observed")

    needed_cols <- c("time_period", "city", "emission", "year", "chill_dayofyear", "model")
    first_frost_limited_obs <- subset(first_frost_limited_obs, select=needed_cols)
    fifth_frost_limited_obs <- subset(fifth_frost_limited_obs, select=needed_cols)

    ##### modeled
    # subset

    first_frost_double_limited <- first_frost_limited %>% filter(model != "Observed")
    fifth_frost_double_limited <- fifth_frost_limited %>% filter(model != "Observed")

    # get medians
    first_frost_double_limited_M <- first_frost_double_limited %>%
                                    group_by(time_period, city, emission, year) %>%
                                    summarise(chill_dayofyear = median(chill_dayofyear)) %>%
                                    data.table()

    fifth_frost_double_limited_M <- first_frost_double_limited %>%
                                    group_by(time_period, city, emission, year) %>%
                                    summarise(chill_dayofyear = median(chill_dayofyear)) %>%
                                    data.table()

    first_frost_double_limited_M$model <- "median_of_19_models"
    fifth_frost_double_limited_M$model <- "median_of_19_models"

    first_frost_double_limited_M <- rbind(first_frost_limited_obs, first_frost_double_limited_M)
    fifth_frost_double_limited_M <- rbind(fifth_frost_limited_obs, fifth_frost_double_limited_M)

    first_frost_double_limited_M_TS <- plot_frost_TS(dt=first_frost_double_limited_M, 
                                                     colname="chill_dayofyear",
                                                     title_ = paste0("First frost day, ", ct))
    
    fifth_frost_double_limited_M_TS <- plot_frost_TS(dt=fifth_frost_double_limited_M, 
                                                     colname="chill_dayofyear",
                                                     title_ = paste0("Fifth frost day, ", ct))

    ggsave(plot = first_frost_double_limited_M_TS, 
           filename = paste0(due, "_1st_frost_median_TS_", ct, , "_", em, ".png"), 
           width = 5.2, height = 3.4, units = "in",
           dpi = 400, device = "png",
           path = out_dir)

    ggsave(plot = fifth_frost_double_limited_M_TS, 
           filename = paste0(due, "_5th_frost_median_TS_", ct, , "_", em, ".png"),
           width = 5.2, height = 3.4, units = "in", 
           dpi = 400, device = "png",
           path = out_dir)

    }
  }
}



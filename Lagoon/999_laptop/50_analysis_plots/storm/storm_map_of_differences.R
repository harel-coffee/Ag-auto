rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/storm/"
plot_dir <- paste0(in_dir, "new_plots/")
           
all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
head(all_storms, 2)

all_storms <- all_storms %>%
              filter(return_period != "2006-2025")%>%
              data.table()

all_storms <- within(all_storms, 
                    remove(five_years, ten_years, 
                           fifteen_years, twenty_years))

########################################################################
clusters <- sort(unique(all_storms$cluster))

clust <- clusters[4]
emissions <- sort(unique(all_storms$emission))

for (clust in clusters){
  curr_dt <- all_storms %>% filter(cluster == clust)
  biased_dt <- storm_diff_4_map_obs_or_modeled(dt_dt=curr_dt, 
                                               diff_from="1979-2016")
  unbiased_dt <- storm_diff_4_map_obs_or_modeled(dt_dt=curr_dt, 
                                                 diff_from="1950-2005")

  biased_dt <- biased_dt %>%
               group_by(location, emission, return_period, cluster) %>% 
               summarise(perc_diff_meds = median(perc_diff)) %>% 
               data.table()

  unbiased_dt <- unbiased_dt %>%
                 group_by(location, emission, return_period, cluster) %>% 
                 summarise(perc_diff_meds = median(perc_diff)) %>% 
                 data.table()

  biased_min <- min(biased_dt$perc_diff_meds)
  biased_max <- max(biased_dt$perc_diff_meds)

  unbiased_min <- min(unbiased_dt$perc_diff_meds)
  unbiased_max <- max(unbiased_dt$perc_diff_meds)

  clr_lim <- ceiling(max(abs(biased_min), abs(biased_max), 
                         abs(unbiased_min), abs(unbiased_max)))

  time_ps <- sort(unique(biased_dt$return_period))
  for (em in emissions){
    for (tp in time_ps){
      plt_dt_bias <- biased_dt %>% 
                     filter(emission == em & return_period==tp) %>% 
                     data.table()

      plt_dt_unbias <- unbiased_dt %>% 
                       filter(emission == em & return_period==tp) %>% 
                       data.table()

      assign(x = paste0("bias_", 
                        gsub("[.]", "", gsub("\ ", "", tolower(em))),
                        "_", 
                        gsub("-", "_", tp)) ,
             value = geo_map_perc_diff(dt_dt = plt_dt_bias,
                                       col_col = "perc_diff_meds", 
                                       color_limit = clr_lim,
                                       ttl = "8",
                                       sub = "biased differences"))
      assign(x = paste0("unbias_", 
                        gsub("[.]", "", gsub("\ ", "", tolower(em))),
                        "_", 
                        gsub("-", "_", tp)) ,
             value = geo_map_perc_diff(dt_dt = plt_dt_unbias,
                                       col_col = "perc_diff_meds", 
                                       color_limit = clr_lim))

    }
  }
}






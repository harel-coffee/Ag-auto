##
##   We are trying to see how the 
##   Giridhar has clustered the locations
##

rm(list=ls())

library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)

library(ggplot2)
library(ggpubr)



data_dir <- file.path("/Users/hn/Documents/01_research_data/my_aeolus_2015/all_local/")
out_dir <- data_dir

f_name <- "CMPOP_hist_85.rds"
dt <- data.table(readRDS(paste0(data_dir, f_name)))

dt <- dt %>%
      filter(year <= 2015) %>%
      data.table()

needed_cols <- c("CountyGroup", "CumDDinC", "CumDDinF", "DailyDD", "day", "dayofyear",
                 "latitude", "longitude", "month", "year")
dt <- subset(dt, select=needed_cols)

dt <- dt %>%
      filter(month == 11 & day == 3) %>%
      data.table()

dt$location <- paste0(dt$latitude, "_", dt$longitude)

# dt$CountyGroup = as.character(dt$CountyGroup)
# dt[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
# dt[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

dt_prep <- within(dt, remove(CountyGroup, DailyDD, day, dayofyear, month, latitude, longitude))

dt_medians <- dt_prep[, .(median_gdd = median(CumDDinF)), by = c("location")]
dt_means <- dt_prep[, .(mean_gdd = mean(CumDDinF)), by = c("location")]


clusters_obj_medians <- kmeans(dt_medians$median_gdd, centers = 2, nstart = 100)
clusters_obj_means <- kmeans(dt_means$mean_gdd, centers = 2, nstart = 100)

clusters_medians = data.table(location = dt_medians$location,
                              median_gdd = dt_medians$median_gdd,
                              cluster_label = clusters_obj_medians$cluster)

clusters_means = data.table(location = dt_means$location,
                            median_gdd = dt_means$mean_gdd,
                            cluster_label = clusters_obj_means$cluster)


dt_old_clusters <- subset(dt, select = c("CountyGroup", "location"))
dt_old_clusters <- unique(dt_old_clusters)

clusters_medians <- merge(clusters_medians, dt_old_clusters)
clusters_means <- merge(clusters_means, dt_old_clusters)

head(clusters_medians, 2)

clusters_medians[CountyGroup == 1]$CountyGroup = 10
clusters_medians[CountyGroup == 2]$CountyGroup = 20

clusters_medians[cluster_label == 1]$cluster_label = 20
clusters_medians[cluster_label == 2]$cluster_label = 10

head(clusters_means, 2)

clusters_means[CountyGroup == 1]$CountyGroup = 10
clusters_means[CountyGroup == 2]$CountyGroup = 20

clusters_means[cluster_label == 1]$cluster_label = 20
clusters_means[cluster_label == 2]$cluster_label = 10


clusters_medians$diff <- clusters_medians$cluster_label - clusters_medians$CountyGroup
clusters_means$diff <- clusters_means$cluster_label - clusters_means$CountyGroup

sum(abs(clusters_medians$diff))
sum(abs(clusters_means$diff))








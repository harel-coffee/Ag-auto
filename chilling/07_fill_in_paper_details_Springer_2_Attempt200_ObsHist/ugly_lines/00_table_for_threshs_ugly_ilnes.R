#
# This is an update of table_for_thresh.R which produces table for plotting the
# stupid Figure. 6 in the google doc paper. It is the ugly plot with several lines
# in it that you do not like.
# March 6, 2020
rm(list=ls())

library(data.table)
library(ggpubr)
library(tidyverse)
library(ggplot2)
########################################################################################
#######                                                                          #######
#######                Add time periods with and without overlap                 #######
#######                                                                          #######
########################################################################################
clean_data <- function(data){
    #
    #  toss modeled historical
    #
    data <- data %>% 
            filter(time_period != "1950-2005") %>%
            data.table()

    data = within(data, remove(sum, sum_J1, sum_F1, sum_M1, sum_A1, 
                               lat, long, chill_season, model, 
                               location, start, year))
    time_periods <- c("1979-2015", "2006-2025", "2026-2050", "2051-2075", "2076-2099")
    data$time_period <- factor(data$time_period, levels = time_periods, order=TRUE)
    data$emission[data$emission=="rcp45"] <- "RCP 4.5"
    data$emission[data$emission=="rcp85"] <- "RCP 8.5"

    data_F <- data %>% 
              filter(time_period %in% c("2006-2025", "2026-2050", "2051-2075", "2076-2099"))%>% 
              data.table()

    data_H <- data %>% 
              filter(time_period %in% c("1979-2015"))%>%
              data.table()

    data_H_45 <- data_H
    data_H_45$emission = "RCP 4.5"
    data_H$emission = "RCP 8.5"

    data <- rbind(data_F, data_H, data_H_45)
    return(data)
}

########################################################################################
###
###
###

data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

########################################################################################
sept_summary <- data.table(readRDS(paste0(data_dir, "sept_summary_comp.rds")))
limited_cities <- read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T)
limited_cities$location <- paste0(limited_cities$lat, "_", limited_cities$long)
limited_cities <- within(limited_cities, remove(lat, long))

#
# Pick up limited cities
#
sept_summary <- sept_summary %>% 
                filter(location %in% limited_cities$location) %>%
                data.table()

sept_summary <- dplyr::left_join(x = sept_summary, y = limited_cities, by = "location")

#
# remove extra columns
#
sept_summary <- clean_data(sept_summary)
Eugene_85_F3 <- sept_summary %>% filter(city %in% c("Eugene") & emission == "RCP 8.5" & time_period == "2076-2099")
Omak_85_F3 <- sept_summary %>% filter(city %in% c("Omak") & emission == "RCP 8.5" & time_period == "2076-2099")

# index_Omak <- data.table(Omak_85_F3 == 0)
# index_Eugene <- data.table(Eugene_85_F3 == 0)

#
# replace zeros with 367 for places that did not meet chill thresholds
#
index <- sept_summary == 0
sept_summary[index] <- 367

#######
#######     Compute Medians
#######
#
# Median is being taken over models. Kill models
#
data_medians <- sept_summary %>% 
                group_by(emission, time_period, city) %>%
                summarise_at(.funs = funs(med = median), vars(thresh_20:thresh_75)) %>%
                data.table()

data_means <- sept_summary %>% 
                group_by(emission, time_period, city) %>%
                summarise_at(.funs = funs(mean = mean), vars(thresh_20:thresh_75)) %>%
                data.table()


data_90th <- sept_summary %>% 
             group_by(emission, time_period, city) %>%
             summarise_all(list(Q90 = quantile), probs = 0.9) %>%
             data.table()
#######
#######     Compute InterQuantile Ranges
#######
data_IQR <- sept_summary %>% 
            group_by(emission, time_period, city) %>%
            summarise_at(.funs = funs(IntQntRng = IQR), vars(thresh_20:thresh_75))%>%
            data.table()
#######
#######     merge medians and IQRs
#######
# data_merged = merge(data_medians, data_IQR, by=c("emission", "time_period", "city"))

#######
#######     Writing Time
#######
write_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/tables/table_for_ugly_lines_plots/"

write.table(data_medians, 
            file = paste0(write_dir, "medians.csv"), 
            row.names = FALSE, 
            col.names = TRUE, 
            sep = ",")


write.table(data_means, 
            file = paste0(write_dir, "means.csv"), 
            row.names = FALSE, 
            col.names = TRUE, 
            sep = ",")


write.table(data_IQR, 
            file = paste0(write_dir, "IQR.csv"), 
            row.names = FALSE, 
            col.names = TRUE, 
            sep = ",")

write.table(data_90th, 
            file = paste0(write_dir, "quan_90.csv"), 
            row.names = FALSE, 
            col.names = TRUE, 
            sep = ",")



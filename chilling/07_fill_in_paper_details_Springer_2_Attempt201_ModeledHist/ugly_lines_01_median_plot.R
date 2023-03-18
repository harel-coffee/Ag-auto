#
# This is a copy of the same file from /Users/hn/Documents/00_GitHub/Ag/chilling/06_fill_in_paper_details
# which includes more scripts!
#
rm(list=ls())

library(data.table)
library(dplyr)
library(ggpubr)
library(ggplot2)
###########################################
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
    data = within(data, remove(sum, sum_J1, sum_F1, sum_M1, sum_A1, 
                               lat, long, chill_season, model, 
                               location, start, year))
    time_periods <- c("1950-2005", "2006-2025", "2026-2050", "2051-2075", "2076-2099")
    data$emission[data$emission=="rcp45"] <- "RCP 4.5"
    data$emission[data$emission=="rcp85"] <- "RCP 8.5"

    data_F <- data %>% 
              filter(time_period %in% c("2006-2025", "2026-2050", "2051-2075", "2076-2099"))%>% 
              data.table()

    data_H <- data %>% 
              filter(time_period %in% c("1950-2005"))%>% # 1979-2015
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
             
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"
DoY_map <- read.csv(paste0(param_dir, "chill_DoY_map.csv"), as.is=TRUE)

data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
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

#######
#######     Compute Medians
#######
#
# Median is being taken over models. Kill models
#
data <- sept_summary %>% 
        group_by(emission, time_period, city) %>%
        summarise_at(.funs = funs(med = median), vars(thresh_20:thresh_75)) %>%
        data.table()

###########################################

#
# clean data
#
setnames(data, old=c("thresh_20_med", "thresh_25_med",
                     "thresh_30_med", "thresh_35_med",
                     "thresh_40_med", "thresh_45_med",
                     "thresh_50_med", "thresh_55_med",
                     "thresh_60_med", "thresh_65_med",
                     "thresh_70_med", "thresh_75_med"), 
               new=c("20", "25", "30", "35", "40", 
                     "45", "50", "55", "60", "65", "70", "75"))

data <- data %>% filter(time_period != "2006-2025") %>% data.table()
data$time_period[data$time_period == "1950-2005"] <- "Historical" # 1979-2015
time_periods = c("Historical", "2026-2050", "2051-2075", "2076-2099")
data$time_period <- factor(data$time_period, levels = time_periods, order=TRUE)

data_melt = melt(data, id=c("city", "emission", "time_period"))

# Convert the column variable to integers
data_melt[,] <- lapply(data_melt, factor)
data_melt[,] <- lapply(data_melt, function(x) type.convert(as.character(x), as.is = TRUE))
time_periods = c("Historical", "2026-2050", "2051-2075", "2076-2099")
data_melt$time_period <- factor(data_melt$time_period, levels = time_periods, order=TRUE)

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

data_melt <- data_melt %>% 
             filter(city %in% ict) %>% 
             data.table()

data_melt$city <- factor(data_melt$city, levels = ict, order=TRUE)

tickSize = 16
axlabelSize = 18
the_thm <- theme(plot.margin = unit(c(t=.2, r=.2, b=.2, l=0.2), "cm"),
                 panel.border = element_rect(fill=NA, size=.3),
                 panel.grid.major = element_line(size = 0.05),
                 panel.grid.minor = element_blank(),
                 panel.spacing = unit(.25, "cm"),
                 legend.position = "bottom", 
                 legend.key.size = unit(1.5, "line"),
                 legend.spacing.x = unit(.05, 'cm'),
                 panel.spacing.y = unit(.5, 'cm'),
                 legend.text = element_text(size=axlabelSize),
                 legend.margin = margin(t=0, r=0, b=0, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 plot.title = element_text(size=axlabelSize, face = "bold"),
                 plot.subtitle = element_text(face = "bold"),
                 strip.text.x = element_text(size=axlabelSize, face="bold"),
                 strip.text.y = element_text(size=axlabelSize, face="bold"),
                 axis.ticks = element_line(size=.1, color="black"),
                 axis.title.x = element_text(size = axlabelSize, face="bold", margin = margin(t=10, r=0, b=0, l=0)),
                 axis.title.y = element_blank(), # element_text(size = axlabelSize, face="bold", margin = margin(t=0, r=10, b=0, l=0)),
                 axis.text.x = element_text(size = tickSize, face="plain", color="black", angle=90, hjust = 1),
                 axis.text.y = element_text(size = tickSize, face="plain", color="black")
                )

color_ord <- c("black", "dodgerblue", "olivedrab4", "tomato1")
color_ord <- c("grey47" , "dodgerblue", "olivedrab4", "red") #

plot_path <- "/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/03_Springer_3/"

if (dir.exists(plot_path) == F) {
  dir.create(path = plot_path, recursive = T)
}

x_start = 35
qual = 400

data_melt$emission <- factor(data_melt$emission, 
                                levels=c("RCP 8.5", "RCP 4.5"),
                                order=TRUE)

plot = ggplot(data_melt, aes(y=variable, x=value), fill=factor(time_period)) + 
       geom_path(aes(colour = factor(time_period))) + 
       facet_grid(~ emission ~ city, scales = "free") + 
       labs(x = "accumulated chill portions", y = "day of year", fill = "Climate Group") +
       scale_color_manual(labels = time_periods, values = color_ord) + 
       scale_x_continuous(breaks = DoY_map$day_count_since_sept, labels= DoY_map$letter_day) + 
       scale_y_continuous(limits = c(20, 75), breaks = seq(20, 80, by = 10)) +
       the_thm + 
       coord_cartesian(xlim = c(min(data_melt$value), max(data_melt$value))) 
plot


plot = ggplot(data_melt, aes(x=variable, y=value), fill=factor(time_period)) + 
       geom_path(aes(colour = factor(time_period))) + 
       facet_grid(~ emission ~ city, scales = "free") + 
       labs(x = "accumulated chill portions", y = "day of year", fill = "Climate Group") +
       scale_color_manual(labels = time_periods, values = color_ord) + 
       scale_y_continuous(breaks = DoY_map$day_count_since_sept, labels= DoY_map$letter_day) + 
       scale_x_continuous(limits = c(x_start, 75), breaks = seq(x_start, 80, by = 10)) +
       the_thm + 
       coord_cartesian(ylim = c(min(data_melt$value), max(data_melt$value))) 
plot


qual = 400
output_name = paste0("median_DoY_thresh_start_", x_start, ".pdf")
ggsave(filename=output_name, plot=plot, device="pdf", 
       path=plot_path, 
       width=12, height=7, unit="in", dpi=qual)

output_name = paste0("median_DoY_thresh_start_", x_start, ".png")
ggsave(filename=output_name, plot=plot, device="png", 
       path=plot_path, 
       width=12, height=7, unit="in", dpi=qual)





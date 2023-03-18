# 
# This is obtained by copying "chill_model_organized_TS(accum_CP)"
# The title and y-labels are wrong. When we do median and group by emission
# location, model. We are not taking median.
#

# 1. Load packages --------------------------------------------------------
rm(list=ls())
library(ggpubr) # library(plyr)
library(tidyverse)
library(data.table)
library(ggplot2)
options(digits=9)
options(digit=9)

##############################################################################
############# 
#############              ********** start from here **********
#############
##############################################################################
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
limited_locs <- read.csv(paste0(param_dir, "limited_locations.csv"), 
                                header=T, sep=",", as.is=T)

main_in_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
write_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/figures/Accum_CP_Sept_Apr/"
if (dir.exists(file.path(write_dir)) == F) { dir.create(path = write_dir, recursive = T)}

summary_comp <- data.table(readRDS(paste0(main_in_dir, "sept_summary_comp.rds")))

summary_comp$location <- paste0(summary_comp$lat, "_", summary_comp$long)
limited_locs$location <- paste0(limited_locs$lat, "_", limited_locs$long)
summary_comp <- summary_comp %>% filter(location %in% limited_locs$location)
summary_comp <- left_join(summary_comp, limited_locs)

######################################################################
#####
#####                Clean data
#####
#######################################################################

summary_comp <- summary_comp %>% filter(time_period != "1950-2005") %>% data.table()
summary_comp <- summary_comp %>% filter(time_period != "1979-2015") %>% data.table()
# summary_comp$emission[summary_comp$time_period=="1979-2015"] <- "Observed"

# summary_comp <- summary_comp %>% filter(emission != "rcp85") %>% data.table()
summary_comp$emission[summary_comp$emission=="rcp45"] <- "RCP 4.5"

summary_comp$emission[summary_comp$emission=="rcp85"] <- "RCP 8.5"

unique(summary_comp$emission)
unique(summary_comp$time_period)

summary_comp <- within(summary_comp, remove(location, lat, long, thresh_20, 
                                            thresh_25, thresh_30, thresh_35,
                                            thresh_40, thresh_45, thresh_50,
                                            thresh_55, thresh_60, thresh_65,
                                            thresh_70, thresh_75, sum_J1, sum_F1, sum_M1,
                                            sum))

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

summary_comp <- summary_comp %>% 
                filter(city %in% ict) %>% 
                data.table()

summary_comp$city <- factor(summary_comp$city, levels = ict, order=TRUE)


omak85 <- summary_comp %>% filter(emission == "RCP 8.5" & city == "Omak")


lw1 <- loess(formula = sum_A1 ~ year, data= omak85, degree = 2)
predictions <- lw1$fitted # or predict(lw1)

omak85$predictions <- predictions

# we need to do the following because for each year
# there are 19 models, i.e. 19 values that are mapped into 1 value 19 times.
# predictions <- unique(predictions)
omak85 <- within(omak85, remove(model, sum_A1))
omak85 <- unique(omak85)


lw1 <- loess(sum_A1 ~ year, data=omak85, degree = 2)
plot(omak85$sum_A1 ~ omak85$year, data=omak85,pch=19,cex=0.1)
j <- order(omak85$year)
lines(omak85$year[j],lw1$fitted[j],col="red",lwd=3)





rm(list=ls())

library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)

data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

sept_summary_comp <- readRDS(paste0(data_dir, "sept_summary_comp.rds")) %>%
                     data.table()

head(sept_summary_comp, 2)
dim(sept_summary_comp)

limited_cities <- read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T)
limited_cities$location <- paste0(limited_cities$lat, "_", limited_cities$long)

sept_summary_comp <- sept_summary_comp %>%
                     filter(location %in% limited_cities$location) %>%
                     data.table()

keep_cols <- c("chill_season", "location", "sum", 
               "sum_A1", "year", "model", "emission", "time_period", 
               "start")

sept_summary_comp <- subset(sept_summary_comp, select=keep_cols)

# sept_summary_comp <- inner_join(x = sept_summary_comp, y = limited_cities, by = "location")
sept_summary_comp <- dplyr::left_join(x = sept_summary_comp, y = limited_cities, by = "location")

sept_summary_comp_observed <- sept_summary_comp %>%
                              filter(model == "observed") %>%
                              data.table()

sept_summary_comp_1979_1980 <- sept_summary_comp %>% 
                               filter(chill_season == "chill_1979-1980" & model == "observed") %>% 
                               data.table()

sept_summary_comp_2014_2015 <- sept_summary_comp %>% 
                               filter(chill_season == "chill_2014-2015" & model == "observed") %>% 
                               data.table()

######
###### 2050
######
sept_summary_comp_2049_2050 <- sept_summary_comp %>% 
                               filter(chill_season == "chill_2049-2050") %>% 
                               data.table()

# A <- aggregate(sept_summary_comp_2049_2050$sum_M1, 
#                 by=list(sept_summary_comp_2049_2050$chill_season,
#                         sept_summary_comp_2049_2050$location,
#                         sept_summary_comp_2049_2050$city, 
#                         sept_summary_comp_2049_2050$emission,
#                         sept_summary_comp_2049_2050$start), 
#                 FUN=min)
A <- sept_summary_comp_2049_2050 %>% 
     group_by(chill_season, location, city, emission, start) %>% 
     slice(which.min(sum_A1)) # or A1?
View(A)

# A <- aggregate(sept_summary_comp_2049_2050$sum_M1, 
#                 by=list(sept_summary_comp_2049_2050$chill_season,
#                         sept_summary_comp_2049_2050$location,
#                         sept_summary_comp_2049_2050$city, 
#                         sept_summary_comp_2049_2050$emission,
#                         sept_summary_comp_2049_2050$start), 
#                 FUN=max)
A <- sept_summary_comp_2049_2050 %>% 
     group_by(chill_season, location, city, emission, start) %>% 
     slice(which.max(sum_A1)) # or A1?
View(A)

######
###### 2079
######
sept_summary_comp_2097_2098 <- sept_summary_comp %>% 
                               filter(chill_season == "chill_2097-2098") %>% 
                               data.table()


A <- sept_summary_comp_2097_2098 %>% 
     group_by(chill_season, location, city, emission, start) %>% 
     slice(which.min(sum_A1))
View(A)


A <- sept_summary_comp_2097_2098 %>% 
     group_by(chill_season, location, city, emission, start) %>% 
     slice(which.max(sum_M1)) # or A1?
View(A)



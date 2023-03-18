library(data.table)
library(dplyr)


dir_1 <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/Feb/"
dir_2 <- "too_much_info/"
dir <- paste0(dir_1, dir_2)
file_name_1 <- "first_frost_till_Feb.rds"

first_frost_till_Feb <- readRDS(paste0(dir, file_name_1))

first_frost_till_Feb <- within(first_frost_till_Feb,
                               remove(month, day, tmin, extended_DoY))

param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
selected_CTs <- read.csv(paste0(param_dir, "limited_locations.csv"), as.is=TRUE)

cities <- c("Hood River", "Walla Walla", "Richland", "Yakima", "Wenatchee", "Omak")
selected_CTs <- selected_CTs %>% filter(city %in% cities)
selected_CTs$location <- paste0(selected_CTs$lat, "_", selected_CTs$long)
 <- within(selected_CTs, remove(lat, long))

first_frost_till_Feb <- first_frost_till_Feb %>% 
                        filter(location %in% selected_CTs$location) %>% 
                        dadta.table()
selected_CTs
first_frost_till_Feb <- merge(first_frost_till_Feb, selected_CTs)

saveRDS(first_frost_till_Feb, paste0(dir_1, "first_frost_till_Feb.rds"))



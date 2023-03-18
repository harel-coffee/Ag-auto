source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/4th_draft/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/4th_draft/chill_plot_core.R"
source(source_path_1)
source(source_path_2)


param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
LOI <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T))

#######################################################################################
# Read Data

data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
data_dir <- paste0(data_dir, due, "/")
DoY_Map <- read.csv("/Users/hn/Documents/GitHub/Ag/DoY_Map.csv", as.is=TRUE)

first <- data.table(readRDS(paste0(data_dir, "first_frost_till_", due, ".rds")))
first <- first %>% filter(year != 1949) %>% data.table()
first <- pick_single_cities_by_location(dt=first, city_info=LOI)

first_medians <- first %>%
                 group_by(time_period, location, emission) %>%
                 summarise(median = median(chill_dayofyear)) %>%
                 data.table()

LOI$location <- paste0(LOI$lat, "_", LOI$long)
LOI <- within(LOI, remove(lat, long))

dim(first_medians)

head(first_medians, 2)

class(first_medians)


first_medians_merged_before <- merge(first_medians, LOI, by="location", all.x=T)
dput(head(first_medians_merged_before, 5))

first_medians <- as.data.frame(first_medians)
first_medians <- data.table(first_medians)
first_medians_merged_after <- merge(first_medians, LOI, by="location", all.x=T)
dput(head(first_medians_merged_after, 5))


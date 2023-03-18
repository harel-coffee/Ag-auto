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

keep_cols <- c("chill_season", "location",
               "sum_A1", "year", "model", "emission", "time_period", 
               "start")

sept_summary_comp <- subset(sept_summary_comp, select=keep_cols)

sept_summary_comp <- sept_summary_comp %>% filter(time_period != "2006-2025")%>%data.table()

# sept_summary_comp <- inner_join(x = sept_summary_comp, y = limited_cities, by = "location")
sept_summary_comp <- dplyr::left_join(x = sept_summary_comp, y = limited_cities, by = "location")

sept_summary_comp_summary <- sept_summary_comp %>%
                             group_by(city, emission, time_period, lat) %>%
                             summarise(median=median(sum_A1)) %>%
                             data.table()

sept_summary_comp_summary <- dcast(sept_summary_comp_summary, city ~ emission + time_period)

##############
## Sort rows
limited_cities <- within(limited_cities, remove("location", "long"))
sept_summary_comp_summary <- dplyr::left_join(x = sept_summary_comp_summary, y = limited_cities, by = "city") %>% data.table()
sept_summary_comp_summary <- sept_summary_comp_summary[order(-lat),]
sept_summary_comp_summary <- within(sept_summary_comp_summary, remove("lat"))
##############

write.csv(sept_summary_comp_summary, 
          file = paste0("/Users/hn/Documents/00_GitHub/", 
                        "Ag_papers/Chill_Paper/tables/", 
                        "CP_medians.csv"),
          row.names=FALSE)



sept_summary_comp_no_model_hist <- within(sept_summary_comp_summary, remove("historical_1950-2005"))

write.csv(sept_summary_comp_no_model_hist, 
          file = paste0("/Users/hn/Documents/00_GitHub/", 
                        "Ag_papers/Chill_Paper/tables/", 
                        "CP_medians_noModeledHistorical.csv"))


##### Computer percentage difference
n_cols <- ncol(sept_summary_comp_no_model_hist)
diffs <- data.frame(sept_summary_comp_no_model_hist)
diffs[, 3:n_cols] <- diffs[, 3:n_cols] - diffs[, 2]
diffs[, 3:n_cols] <- (diffs[, 3:n_cols] / diffs[, 2]) * 100
diffs <- data.table(diffs)

write.csv(diffs, 
          file = paste0("/Users/hn/Documents/00_GitHub/", 
                        "Ag_papers/Chill_Paper/tables/", 
                        "CP_perc_change_observed.csv"))

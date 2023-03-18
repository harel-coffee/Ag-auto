library(data.table)
library(dplyr)


data_dir_base <- "/Users/hn/Documents/01_research_data/cold_hardiness/"

sub_dirs <- c("cabernet_sauvignon", "chardonnay", "merlot", "reisling")

all_data <- data.table()

for (sub in sub_dirs){
  a_variety_data <- data.table()
  grape_type <- sub
  curr_dir <- paste0(data_dir_base, sub, "/")

  list_of_files = list.files(path = curr_dir, pattern = ".csv")

  for (a_file in list_of_files){
    a_file <- read.csv(paste0(curr_dir, a_file), as.is=TRUE)
    all_data <- rbind(all_data, a_file)
    a_variety_data <- rbind(a_variety_data, a_file)
  }
  a_variety_data$model[a_variety_data$model == "historical"] <- "obs_hist"
  a_variety_data$location <- paste(a_variety_data$lat, a_variety_data$long)
  # a_variety_data$range[a_variety_data$range == "1979-2020"] <- "1979-2016"
  
  write.csv(a_variety_data, 
          paste0(data_dir_base, sub, "_CH_stats_by_loc.csv"), row.names = F)
}

all_data$model[all_data$model == "historical"] <- "obs_hist"
# all_data$range[all_data$range == "1979-2020"] <- "1979-2016"
all_data$location <- paste(all_data$lat, all_data$long)
  
write.csv(all_data, 
          paste0(data_dir_base, "CH_stats_by_loc.csv"), row.names = F)
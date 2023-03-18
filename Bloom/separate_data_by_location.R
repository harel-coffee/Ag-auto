library(data.table)
library(dplyr)

##################################################################

in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/bloom/"
param_dir <- "/Users/hn/Documents/GitHub/Ag/Bloom/parameters/"

##################################################################

frost_out_dir <- paste0(in_dir, "frost/")
bloom_out_dir <- paste0(in_dir, "bloom/")
CP_out_dir <- paste0(in_dir,"CP/")

##################################################################
if (dir.exists(file.path(frost_out_dir)) == F) {
  dir.create(path = file.path(frost_out_dir), recursive=T)
}

if (dir.exists(file.path(bloom_out_dir)) == F) {
  dir.create(path = file.path(bloom_out_dir), recursive=T)
}

if (dir.exists(file.path(CP_out_dir)) == F) {
  dir.create(path = file.path(CP_out_dir), recursive=T)
}

##################################################################
locations <- read.csv(paste0(param_dir, "file_list.txt"), 
                      as.is=TRUE, header=FALSE)
locations <- as.vector(locations$V1)
locations <- gsub("data_", "", locations)

CP_dt <- readRDS(paste0(in_dir, "sept_summary_comp.rds")) %>%
             data.table()

frost_dt <- readRDS(paste0(in_dir, "first_frost.rds")) %>%
            data.table()

bloom_dt <- readRDS(paste0(in_dir, "fullbloom_50percent_day.rds")) %>%
         data.table()
bloom_dt$location <- paste0(bloom_dt$lat, "_", bloom_dt$long)


for (loc in locations){
    curr_frost <- frost_dt %>% filter(location == loc) %>% data.table()
    curr_CP <- CP_dt %>% filter(location == loc) %>% data.table()
    curr_bloom <- bloom_dt %>% filter(location == loc) %>% data.table()

    saveRDS(object = curr_frost,
            file=paste0(frost_out_dir, loc, ".rds"))

    saveRDS(object = curr_CP,
            file=paste0(CP_out_dir, loc, ".rds"))

    saveRDS(object = curr_bloom,
            file =paste0(bloom_out_dir, loc, ".rds"))
}




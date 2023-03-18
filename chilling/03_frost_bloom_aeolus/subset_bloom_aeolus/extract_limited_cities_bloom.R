########################################################################
#
# We are trying here to extract the 10 locations chosen for
# chill project stuff, to plot the bloom time for them.
# 
########################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)

source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)

#########
######### Running this is gonna take at least 40 minutes, 
######### IF it is successful! R memory allocation problem!
#########
data_dir <- "/data/hydro/users/Hossein/codling_moth_new/local/processed/vertdd_with_new_normal_params/"
out_dir <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/bloom_limited_cities/"
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"

if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = out_dir, recursive = T)
}

cities <- read.csv(paste0(param_dir, "bloom_limited_cities.csv"), as.is=T)
cities <- within(cities, remove(lat, long))
print ("line 32")
print (dim(cities))

emissions <- c("rcp85", "rcp45")
all_dt <- data.table()
for (em in emissions){
  data_dt <- readRDS(paste0(data_dir, "vertdd_combined_CMPOP_", em, ".rds"))
  print (head(data_dt, 2))
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$latitude, "_",data_dt$longitude)
  }
  print (head(data_dt, 2))
  data_dt <- subset(data_dt, 
                    select = c("location", "ClimateGroup", "ClimateScenario", 
                               "year", "dayofyear", 
                               "cripps_pink", "gala", "red_deli"))

  data_dt <- data_dt %>% filter(location %in% cities$location)%>% data.table()
  print(unique(data_dt$location))

  setnames(data_dt, 
           old=c("ClimateGroup", "ClimateScenario"), 
           new=c("time_period", "model"))
  if (em=="rcp85"){
      data_dt$emission <- "RCP 8.5"
     } else {
    	data_dt$emission <- "RCP 4.5"
  }
  all_dt <- rbind(all_dt, data_dt)
}
all_dt <- merge(all_dt, cities)

all_dt$model[all_dt$model == "historical"] <- "observed"
all_dt <- within(all_dt, remove(location))
saveRDS(all_dt, paste0(out_dir, "bloom_limited_cities.rds"))










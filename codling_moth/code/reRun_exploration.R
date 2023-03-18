library(data.table)
library(dplyr)

data_dir <- "/Users/hn/Documents/01_research_data/codling_moth/data_from_Aeolus_reRun_detection/"

A <- readRDS(paste0(data_dir, "combined_CM_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

A$location <- paste0(A$latitude, "_", A$longitude)
a_location <- unique(A$location)[1]

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)

############################################################

A <- readRDS(paste0(data_dir, "bloom_rcp85_1_new.rds")) %>%
     data.table()

View(sort(colnames(A)))

A$location <- paste0(A$latitude, "_", A$longitude)

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)

############################################################

A <- readRDS(paste0(data_dir, "cumdd_CMPOP_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))
############################################################

A <- readRDS(paste0(data_dir, "diapause_map1_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

A$location <- paste0(A$latitude, "_", A$longitude)

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)


############################################################

A <- readRDS(paste0(data_dir, "diapause_rel_data_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

############################################################

A <- readRDS(paste0(data_dir, "diapause_abs_data_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))
############################################################

A <- readRDS(paste0(data_dir, "diapause_plot_data_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

############################################################

A <- readRDS(paste0(data_dir, "generations_Aug_combined_CMPOP_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

A$location <- paste0(A$latitude, "_", A$longitude)

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)

############################################################

A <- readRDS(paste0(data_dir, "combined_CMPOP_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

A$location <- paste0(A$latitude, "_", A$longitude)

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)

############################################################

A <- readRDS(paste0(data_dir, "vertdd_combined_CMPOP_rcp45.rds")) %>%
     data.table()

View(sort(colnames(A)))

unique(A$ClimateScenario)

A$location <- paste0(A$latitude, "_", A$longitude)

A <- A %>% 
     filter(location == a_location) %>%
     data.table()

unique(A$ClimateScenario)














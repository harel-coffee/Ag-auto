.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)



options(digit=9)
options(digits=9)

######################################################################
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

limited_cities <- read.csv(paste0(param_dir, "limitedLocs4Paper.csv"), 
                           as.is=T) %>% data.table()

limited_cities <- within(limited_cities, remove("lat", "long"))

input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/for_paper/"
if (dir.exists(write_dir) == F) {dir.create(path = write_dir, recursive = T)}

files <- c("combined_CM_rcp85", "combined_CM_rcp45", 
           "diapause_rel_data_rcp45",
           "pre_diap_plot_rcp85", "diapause_abs_data_rcp45", 
           "diapause_rel_data_rcp85",
           "bloom_rcp45", "diapause_abs_data_rcp85",
           "bloom_rcp85", "diapause_map1_rcp45", 
           "diapause_map1_rcp85", "pre_diap_plot_rcp45",
           "diapause_plot_data_rcp45",
           "diapause_plot_data_rcp85", 
           "generations_Nov_combined_CMPOP_rcp45", 
           "generations_Nov_combined_CMPOP_rcp85", 
           "generations_Aug_combined_CMPOP_rcp85", 
           "generations_Aug_combined_CMPOP_rcp45",
           "combined_CMPOP_rcp45", "combined_CMPOP_rcp85", 
           "vertdd_combined_CMPOP_rcp45", "vertdd_combined_CMPOP_rcp85")

for (a_file in files){

    data <- data.table(readRDS(paste0(input_dir, a_file, ".rds")))

    if (!("location" %in% colnames(data))){
        data$location <- paste0(data$latitude, "_", data$longitude)
    }

    data <- data %>%
            filter(location %in% limited_cities$location) %>%
            data.table()

    data <- left_join(data, limited_cities) %>% data.table()

    if ("X" %in% colnames(data)){
      data <- within(data, remove(X))
    }
    write.csv(data, 
              file = paste0(write_dir, a_file, ".csv"))
    
}

#######
####### cumdd_data
#######

input_dir <- "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/cumdd_data/"
write_dir <- "/data/hydro/users/Hossein/codling_moth_new/for_paper/cumdd_data/"
if (dir.exists(write_dir) == F) {dir.create(path = write_dir, recursive = T)}

files <- c("cumdd_CMPOP_rcp45", "cumdd_CMPOP_rcp85")

for (a_file in files){

    data <- data.table(readRDS(paste0(input_dir, a_file, ".rds")))
    data$location <- paste0(data$latitude, "_", data$longitude)
    data <- data %>%
            filter(location %in% limited_cities$location) %>%
            data.table()

    data <- left_join(data, limited_cities)%>%
            data.table()

    if ("X" %in% colnames(data)){
      data <- within(data, remove(X))
    }

    write.csv(data, 
              file = paste0(write_dir, a_file, ".csv"))
    
}

####### 
#######  section_46_Pest
####### 

input_dir <- "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/section_46_Pest/"
write_dir <- "/data/hydro/users/Hossein/codling_moth_new/for_paper/section_46_Pest/"
if (dir.exists(write_dir) == F) {dir.create(path = write_dir, recursive = T)}

files <- c("1000F_combined_CMPOP_rcp45",
           "1000F_combined_CMPOP_rcp85",
           "all_1000F_14_days_window_45",
           "all_1000F_14_days_window_85",
           "400F_2nd_combined_CMPOP_rcp85",
           "400F_2nd_combined_CMPOP_rcp45", 
           "all_14_days_window_45",
           "all_14_days_window_85", 
           "start_end_1000F_85", 
           "start_end_1000F_45",
           "start_end_45",
           "start_end_85")

for (a_file in files){
    data <- data.table(readRDS(paste0(input_dir, a_file, ".rds")))
    data$location <- paste0(data$latitude, "_", data$longitude)
    data <- data %>%
            filter(location %in% limited_cities$location) %>%
            data.table()
    data <- left_join(data, limited_cities) %>%
            data.table()

    if ("X" %in% colnames(data)){
      data <- within(data, remove(X))
    }
    
    write.csv(data, 
              file = paste0(write_dir, a_file, ".csv"))
    
}



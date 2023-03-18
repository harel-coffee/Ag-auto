
library(tidyverse)


spatial <- readRDS("data/pruett/RDS/spatial.rds")
files <- list.files("data/pruett/runoff/historical/monthly_summaries/", pattern = "runoff.*.xyz")


read_data <- function(file, variable){

  read.table(paste0("data/pruett/runoff/historical/monthly_summaries/", file), col.names = c("lat", "lon", variable)) %>% 
    filter(lat %in% spatial$lat & lon %in% spatial$lng) %>% 
    mutate(month = str_sub(file, -17, -15),
           group = str_sub(file, -13, -5))

}

read_data <- function(file, variable){

  read.table(paste0("data/pruett/", variable ,"/historical/monthly_summaries/", file), col.names = c("lat", "lon", variable)) %>% 
    filter(lat %in% spatial$lat & lon %in% spatial$lng) %>% 
    mutate(month = str_sub(file, -7, -5),
           group = "hist")

}

runoff <- map(files, read_data, "runoff") %>% bind_rows() 
baseflow <- map(files, read_data, "baseflow") %>% bind_rows() 

df <- merge(baseflow, runoff)

write_rds(df, "data/pruett/runoff/historical/data.RDS")

summary_df <- df %>% group_by(lat, lon) %>% 
  summarise(sum_combined = sum(runoff + baseflow))

write_rds(summary_df, "data/pruett/runoff/hydro_spatial.RDS")


bind_rows(read_rds(paste0("data/pruett/runoff/ccsm3_A1B/data.RDS")), 
      read_rds(paste0("data/pruett/runoff/historical/data.RDS")))


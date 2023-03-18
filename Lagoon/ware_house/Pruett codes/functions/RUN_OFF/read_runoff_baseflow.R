library(tidyverse)
library(stringr)
library(rgdal)
library(sp)

# Load Counties
skagit <- readOGR("geo/Skagit.geo.json") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties data ####
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)

# Load files list
files_df <- tibble(files = list.files("data/pruett/runoff/raw/hadcm_B1/fluxes_monthly_summary/")) %>% 
  mutate(lat_lon = str_sub(files, 8, -5)) %>% 
  separate(lat_lon, c("lat", "lng"), "_", convert = TRUE)

# Set coordinates of files
coordinates(files_df) <- c('lng', 'lat')
proj4string(counties) <- proj4string(files_df)

# subset files
files_df <- files_df[counties,] %>% as.tibble()




convert_runoff <- function(file){
  
  path <- paste0("data/pruett/runoff/raw/hadcm_B1/fluxes_monthly_summary/", file)
  
  df <- read_table2(path, col_names = FALSE) %>% 
    select(year = X1, month = X2, runoff = X12, baseflow = X13) %>% 
    mutate(runoff = runoff + 9999,
           baseflow = baseflow + 9999)
  
  save_path = paste0("data/pruett/runoff/RDS/hadcm_B1/", str_sub(file, 1, -5), ".RDS")
  saveRDS(df, save_path)
}

lapply(files_df$files, convert_runoff)

### read historical and summarise

files_df <- tibble(files = list.files("data/pruett/runoff/RDS/historical/"),
                   file_path = list.files("data/pruett/runoff/RDS/historical/", full.names = TRUE)) %>% 
  mutate(lat_lon = str_sub(files, 8, -5)) %>% 
  separate(lat_lon, c("lat", "lng"), "_", convert = TRUE) %>% 
  mutate(max_combined = map(file_path, summarize_runoff)) %>% 
  select(-file_path)
  
saveRDS(files_df, "data/pruett/runoff/hydro_spatial.RDS")
  
summarize_runoff <- function(file_path){
  df <- readRDS(file_path)
  max(df$runoff + df$baseflow)
}



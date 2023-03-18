
proj4string(counties) <- proj4string(map_df) # Sync Coordinate Systems

# Set spatial coordinates of points
coordinates(map_df) <- c('lng', 'lat')

# Subset data frame by values within counties
map_df <- map_df[counties,] %>% as.tibble()

file_path <- "aeolus/data/historical/"


map_df <- data.frame(file_name = list.files("aeolus/data/historical/")) %>%
  mutate(lat = as.numeric(str_sub(file_name, 6, 13)),
         lng = as.numeric(str_sub(file_name, 15)))

map_df <- data.frame(file_name = list.files(file_path))
  
map(map_df$file_name, RDS_convert)
  

max_precip <- function(file_list){
  
  calc_precip <- function(file_name){
    df <- read_binary(file_name)
    max(df$precip)
  }
  
  file_list %>% 
    mutate(max_precip = map(file_name, calc_precip))
  
}

source("functions/read_binary_8.R")

RDS_convert <- function(file_name, file_path){
  df <- read_binary_(file_name, file_path, FALSE)
  saveRDS(df, paste0("data/pruett/RDS/CanESM2/rcp85/", file_name,".rds"))
}

df <- tibble(file_name = list.files("data/pruett/aeolus/CanESM2/rcp85/"),
             file_path = list.files("data/pruett/aeolus/CanESM2/rcp85/", 
             full.names = TRUE))

map2(df$file_name, df$file_path, RDS_convert)


RDS_convert(df$file_name[1], df$file_path[1])

test <- read_binary_(df$file_name[1], df$file_path[1], TRUE)



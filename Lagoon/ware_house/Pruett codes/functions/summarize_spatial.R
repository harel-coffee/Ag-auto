library(tidyverse)
library(stringr)

df <- tibble(file_name = list.files("data/pruett/dry_days/monthly_prob_octmar/"))

summarize_spatial <- function(file_name){
  read_rds(paste0("data/pruett/dry_days/monthly_prob_octmar/", file_name)) %>% 
    group_by(group, climate_proj, exceedance) %>% 
    summarise(prob_median = first(prob_median))
}

df <- df %>% mutate(lat_lon = str_sub(file_name, 6, -5)) %>%
  separate(lat_lon, c("lat", "lng"), sep = "_", convert = TRUE) %>% 
  mutate(data = map(file_name, summarize_spatial)) %>% 
  unnest()

write_rds(df, "data/pruett/dry_days/spatial_prob.rds")

library(dplyr)
library(tidyr)
library(readr)
library(parallel)

summarize_precip <- function(file_name){
  read_rds(file_name) %>% 
    # mutate(date_time = ymd(paste(year, month, day, sep = "-")),
    #        water_year = year(date_time %m+% months(3))) %>% 
    # filter(!is.na(group)) %>% 
    group_by(year, month) %>% 
    summarize(precip = sum(precip, na.rm = TRUE))
    # ungroup() %>% 
    # group_by(climate_proj, group, model, month) %>% 
    # summarise(precip = mean(precip, na.rm = TRUE))
}

df <- read_rds("spatial.rds") %>% 
  mutate(file_name = paste0("/data/hydro/users/mpruett/historical/", file_name, ".rds")) %>% 
  select(file_name, lat, lng)

df %>% mutate(data = mclapply(file_name, summarize_precip, mc.cores = 64)) %>% 
  unnest() %>% 
  write_rds("precip_summary_hist.rds")


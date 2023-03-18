library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(lubridate)
library(parallel)
library(zoo)

summarize_prob <- function(file_path){
  df <- read_rds(file_path) %>% 
    mutate(time_stamp = ymd(paste(year, month, day, sep="-")),
           water_year = year(time_stamp %m+% months(3)))
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    group_by(group, month, model, year, climate_proj) %>% 
    summarise(precip = sum(precip)) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, month, model, year, climate_proj) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group, model, climate_proj) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
    gather(exceedance, prob, -model, -group, -climate_proj) %>% 
    group_by(climate_proj, group, exceedance) %>% 
    summarise(prob_median = median(prob))
  
  return(octmar_exceedance_val)
  
}

df <- read_rds("spatial.rds") %>% 
  mutate(file_path = paste0("/data/hydro/users/mpruett/data/", file_name, ".rds")) %>% 
  mutate(data = mclapply(file_path, summarize_prob, mc.cores = 32)) %>% 
  unnest()

write_rds(df, "spatial_prob_monthly.rds")

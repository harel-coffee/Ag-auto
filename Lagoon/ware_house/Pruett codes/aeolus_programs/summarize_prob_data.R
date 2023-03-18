library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(parallel)
library(zoo)


summarize_prob <- function(file_name){
  
  file_path = paste0("/data/hydro/users/mpruett/data/", file_name, ".rds")
  
  df <- read_rds(file_path) %>% 
    filter(month >= 10 | month <= 3) %>% 
    group_by(group, climate_proj, model, month, year) %>%
    summarize(precip = sum(precip))
  
  df_monthly_exceedance <- df %>% 
    filter(group == 'hist') %>% 
    group_by(month) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly <- full_join(df, df_monthly_exceedance) %>% 
    filter(!is.na(group))
  
  monthly_exceedance_val <- df_monthly %>% 
    filter(!is.na(climate_proj)) %>% 
    group_by(group, model, climate_proj) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95)))) %>%  
    gather(exceedance, prob, -model, -group, -climate_proj) %>% 
    group_by(group, exceedance, climate_proj) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05))
  
  write_rds(monthly_exceedance_val, path = paste0("/data/hydro/users/mpruett/monthly_prob_month/", file_name, ".rds"))
  
}

df <- read_rds("spatial.rds") 

mclapply(as.character(df$file_name), summarize_prob, mc.cores = 64)



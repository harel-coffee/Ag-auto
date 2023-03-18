library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(parallel)
library(zoo)


summarize_prob <- function(file_name){
  
  file_path = paste0("/data/hydro/users/mpruett/data/", file_name, ".rds")
  
  df <- df %>% 
    group_by(group, climate_proj, model) %>% 
    filter(precip <= quantile(precip, 0.05, na.rm = TRUE)) %>% 
    group_by(group, climate_proj, model, month, year) %>%
    summarize(dry_days = n())
  
  df_monthly_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>% 
    group_by(month) %>% 
    mutate(prob = rank(dry_days)/(n()+1)) %>%
    summarise(dry_days_80 = nth(dry_days, which.min(abs(prob-0.8))),
              dry_days_90 = nth(dry_days, which.min(abs(prob-0.9))),
              dry_days_95 = nth(dry_days, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
    filter(!is.na(group))
  
  monthly_exceedance_val <- df_monthly_prob %>% 
    filter(group != "hist", month >= 10 | month <= 3) %>% 
    group_by(group, model, climate_proj, month) %>%
    mutate(prob = rank(-dry_days)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(dry_days-df_monthly_exceedance$dry_days_80))),
              prob_90 = nth(prob, which.min(abs(dry_days-df_monthly_exceedance$dry_days_90))),
              prob_95 = nth(prob, which.min(abs(dry_days-df_monthly_exceedance$dry_days_95)))) %>%  
    gather(exceedance, prob, -model, -group, -climate_proj, -group, -month) %>% 
    group_by(group, exceedance, climate_proj, month) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05))
  
  write_rds(monthly_exceedance_val, path = paste0("/data/hydro/users/mpruett/dry_days/monthly_prob_monthly/", file_name, ".rds"))
  
}

df <- read_rds("spatial.rds") 

mclapply(as.character(df$file_name), summarize_prob, mc.cores = 16)



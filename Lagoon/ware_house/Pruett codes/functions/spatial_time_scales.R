library(tidyverse)


daily <- read_rds("data/pruett/precip/spatial_prob_daily.rds") %>% 
  as_tibble() %>% 
  select(-max_precip, -file_path) %>% 
  mutate(time_scale = "day")

weekly <- read_rds("data/pruett/precip/spatial_prob_weekly.rds") %>% 
  as_tibble() %>% 
  select(-max_precip, -file_path) %>% 
  mutate(time_scale = "week")

monthly <- read_rds("data/pruett/precip/spatial_prob_monthly.rds") %>% 
  as_tibble() %>% 
  select(-max_precip, -file_path) %>% 
  mutate(time_scale = "month")

bind_rows(daily, weekly, monthly) %>% 
  mutate(file_name = as.character(file_name)) %>% 
  mutate(exceedance_val = case_when(exceedance == "prob_80" ~ 0.2,
                                    exceedance == "prob_90" ~ 0.1,
                                    exceedance == "prob_95" ~ 0.05),
         prob_median = prob_median - exceedance_val) %>% 
  select(-exceedance_val) %>% 
  write_rds("data/pruett/precip/spatial_prob_time.rds")

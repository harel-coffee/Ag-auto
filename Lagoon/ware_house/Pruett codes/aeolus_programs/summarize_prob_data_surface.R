library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(lubridate)
library(stringi)

# combine surface flow data #

files <- list.files("data/pruett/surface/RDS/historical/")
models <- list.dirs("data/pruett/surface/RDS", recursive = FALSE, full.names = FALSE)

read_surface <- function(model, climate_proj, file_name){
  if (model ==  "historical") {
    readRDS(paste0("data/pruett/surface/RDS/", model, "/", file_name))
  } else {
    readRDS(paste0("data/pruett/surface/RDS/", model, "_", climate_proj, "/", file_name))
  }
}

df <- crossing(models, files) %>% 
  mutate(models = stri_replace_last(models, "-", regex = "_")) %>% 
  separate(models, c("model", "climate_proj"), sep = "-", remove = TRUE, extra = "merge", fill = "right") %>% 
  mutate(data = pmap(list(model, climate_proj, files), read_surface)) %>% 
  unnest() %>% 
  group_by(files) %>% 
  nest()



summarize_prob <- function(df){
  
  df_hist <- df %>% filter(model == "historical") %>% 
    mutate(group = "hist")
  
  df_2040 <- df %>% filter(year >= 2025, year <= 2055) %>%
    mutate(group = "2040s")
  
  df_2060 <- df %>% filter(year >= 2045, year <= 2075) %>%
    mutate(group = "2060s")
  
  df_2080 <- df %>% filter(year >= 2065, year <= 2095) %>%
    mutate(group = "2080s")
  
  df <- bind_rows(df_hist, df_2040, df_2060, df_2080) %>%
    mutate(group = as.factor(group))
  
  df <- df %>% 
    mutate(time_stamp = ymd(paste(year, month, "01", sep="-")),
           water_year = year(time_stamp %m+% months(3)),
           combined = runoff + baseflow)
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(combined)/(n()+1)) %>%
    summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
              combined_90 = nth(combined, which.min(abs(prob-0.9))),
              combined_95 = nth(combined, which.min(abs(prob-0.95))))
  
  octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, model, climate_proj) %>%
    mutate(prob = rank(-combined)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_80))),
              prob_90 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_90))),
              prob_95 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_95)))) %>% 
    gather(exceedance, prob, -model, -group, -climate_proj) %>% 
    group_by(climate_proj, group, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05))
  
  
    return(octmar_exceedance_val)
  
}

df_octmar <- df %>% mutate(data = map(data, summarize_prob))

df_octmar <- df_octmar %>% group_by(files) %>% nest()

df_octmar %>% mutate(file_path = paste0("data/pruett/surface/daily_prob_octmar/", files),
                     finished = walk2(data, file_path, write_rds))


read_rds("spatial.rds") %>% 
  mutate(file_path = paste0("/data/hydro/users/mpruett/data/", file_name, ".rds")) %>% 
  mutate(data = walk(file_path, summarize_prob)) %>% 
  unnest() %>% write_rds("spatial_prob.rds")

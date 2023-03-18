library(tidyverse)
library(stringi)

files <- list.files("data/pruett/runoff/RDS/historical/")
models <- list.dirs("data/pruett/runoff/RDS", recursive = FALSE, full.names = FALSE)

read_runoff <- function(model, climate_proj, file_name){
  if (model ==  "historical") {
    readRDS(paste0("data/pruett/runoff/RDS/", model, "/", file_name))
  } else {
    readRDS(paste0("data/pruett/runoff/RDS/", model, "_", climate_proj, "/", file_name))
  }
}

df <- crossing(models, files) %>% 
  mutate(models = stri_replace_last(models, "-", regex = "_")) %>% 
  separate(models, c("model", "climate_proj"), sep = "-", remove = TRUE, extra = "merge", fill = "right") %>% 
  mutate(data = pmap(list(model, climate_proj, files), read_runoff)) %>% 
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
    summarise(prob_median = median(prob))
  
  return(octmar_exceedance_val)
  
}

df_octmar <- df %>% mutate(data = map(data, summarize_prob)) %>% unnest()

df_octmar <- df_octmar %>% mutate(lat_lon = str_sub(files, 8, -5)) %>% 
  separate(lat_lon, c("lat", "lng"), sep = "_", convert = TRUE)

write_rds(df_octmar, "data/pruett/spatial_prob_surface.rds")




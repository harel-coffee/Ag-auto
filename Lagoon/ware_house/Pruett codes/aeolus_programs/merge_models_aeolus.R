library(dplyr)
library(purrr)
library(readr)
library(tidyr)
library(tibble)

source("scripts/read_binary_aeolus.R")

merge_models <- function(file_name){
  
  models <- list.dirs("/data/hydro/jennylabcommon2/metdata/VIC_ext_maca_v2_binary_westUSA/", recursive = FALSE, full.names = FALSE) %>% 
    append("historical")
  
  climate_proj <- c("rcp45", "rcp85")
  
  build_path <- function(model, climate_proj){
    if (model == "historical"){
      paste0("/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/")
    }
    else {
      paste0("/data/hydro/jennylabcommon2/metdata/VIC_ext_maca_v2_binary_westUSA/", model, "/", climate_proj, "/")
    }
  } 
  
  df <- crossing(models, climate_proj) %>% 
    mutate(file_path = map2_chr(models, climate_proj, build_path),
           file_name = as.character(file_name))
  pmap_dfr(list(df$file_name, df$file_path, df$models, df$climate_proj), read_binary) %>% 
    write_rds(paste0("data/", file_name, ".rds")) 
  
}


d <- read_rds("spatial.rds") %>% as_tibble() %>% mutate(file_name = as.character(file_name))
walk(d$file_name, merge_models)






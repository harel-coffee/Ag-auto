######################################################################
rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

options(digit=9)
options(digits=9)
######################################################################
find_county_name <- function(dt_df, curr_fips){
  dt_df <- dt_df %>% filter(fips == curr_fips)
  county_name <- dt_df$st_county
  county_name <- tolower(strsplit(as.character(county_name), "_")[[1]][2])
  return(county_name)
}

find_unique_county_to_county_freq <- function(dt, usa_county_fips){
  # some locations at some year may be novel! drop them
  dt <- na.omit(dt)

  ## add county fips to the data which only has the location of local sites
  dt_merged <- left_join(dt, usa_county_fips, all.x=T, by=c("query_loc" = "location")) %>% data.table()
  dt_merged <- within(dt_merged, remove(query_loc, query_year, st_county))

  setnames(dt_merged, old=c("fips"), new=c("query_fips"))
  setcolorder(dt_merged, c("query_fips", "analog_fips", "Freq", "ClimateScenario"))

  dt_aggregation <- dt_merged[, .(freq = sum(Freq)), by = c("query_fips", "analog_fips", "ClimateScenario")]

  # separate models, perhaps to compare them later:
  dt_agg_bcc_m <- dt_aggregation %>% filter(ClimateScenario == "bcc-csm1-1-m")
  dt_agg_BNU <- dt_aggregation %>% filter(ClimateScenario == "BNU-ESM")
  dt_agg_CanESM2 <- dt_aggregation %>% filter(ClimateScenario == "CanESM2")
  dt_agg_CNRM_CM5 <- dt_aggregation %>% filter(ClimateScenario == "CNRM-CM5")
  dt_agg_GFDLG <- dt_aggregation %>% filter(ClimateScenario == "GFDL-ESM2G")
  dt_agg_GFDLM <- dt_aggregation %>% filter(ClimateScenario == "GFDL-ESM2M")
  return(list(dt_aggregation, dt_agg_bcc_m, dt_agg_BNU, dt_agg_CanESM2, dt_agg_CNRM_CM5, dt_agg_GFDLG, dt_agg_GFDLM))
}


######################################################################
####
####         Set up directories
####
######################################################################

data_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/w_gen_w_prec/500/"
param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

######################################################################
####
####           global Files
####
######################################################################
local_cnty_fips <- "local_county_fips.csv"
usa_cnty_fips <- "all_us_1300_county_fips_locations.csv"

local_cnty_fips <- data.table(read.csv(paste0(param_dir, local_cnty_fips), header=T, sep=",", as.is=T))
usa_cnty_fips <- data.table(read.csv(paste0(param_dir, usa_cnty_fips), header=T, sep=",", as.is=T))
local_fips <- unique(local_cnty_fips$fips)
########################################################################
####
####           Find frequencies at the county level
####
########################################################################
file_pref <- "all_close_analogs_unique_"
time_periods <- c("2051_2075") # , "2051_2075", "2076_2095"
emission_types = c("rcp45", "rcp85")
files_post = ".rds"

############################################################
###
###               Load needed data for plotting
###
############################################################

emission = emission_types[1]

for (time in time_periods){
  
  file <- paste0(file_pref, time, "_", emission, files_post)
  dt_H <- data.table(readRDS(paste0(data_dir, file)))

  # Locations are factors. Kill it.
  dt_H$query_loc <- as.character(dt_H$query_loc)
  output <- find_unique_county_to_county_freq(dt_H, usa_cnty_fips)
  # dt_aggregation <- output[[1]] # all models are here
  # dt_agg_bcc_m <- output[[2]]
  # dt_agg_BNU  <- output[[3]]
  # dt_agg_CanESM2 <- output[[4]]
  # dt_agg_CNRM_CM5 <- output[[5]]
  # dt_agg_GFDLG <- output[[6]]
  # dt_agg_GFDLM <- output[[7]]

  for (ii in 2:7){ # run for different models
    curr_data <- output[[ii]]
    for (curr_fip in local_fips){ # run for different local (future) counties
      one_county_one_model <- curr_data %>% filter(query_fips==curr_fip)
      ######################################################################
      # if the place is novel then the above data is empty, so, no map.
      ######################################################################
      model_name <- unique(one_county_one_model$ClimateScenario)

      data(county.fips) # Load the county.fips dataset for plotting
      cnty <- map_data("county") # Load the county data from the maps package
      cnty2 <- cnty %>%
               mutate(polyname = paste(region, subregion, sep=",")) %>%
               left_join(county.fips, by="polyname")

      cnty2_one_county_one_model <- left_join(one_county_one_model, cnty2, by=c("analog_fips" = "fips"))
      cnty_name <- find_county_name(usa_cnty_fips, curr_fip)
      
      target_annotation <- cnty2
      target_annotation <- target_annotation %>% filter(fips==curr_fip)
      target_annotation <- target_annotation %>%
                           group_by(subregion, region, polyname, group) %>%
                           summarise_at(vars(long, lat), funs(mean(., na.rm=TRUE))) %>%
                           data.table()

      curr_plot <- ggplot(cnty2_one_county_one_model, aes(long, lat, group = group)) + 
                   geom_polygon(data = cnty2, fill="lightgrey") + 
                   geom_polygon(aes(fill = freq), colour = rgb(1, 1, 1, 0.2))  +
                   geom_text(data=target_annotation, aes(long, lat, label = subregion), size=2, fontface="bold", color="red") + 
                   coord_quickmap() + 
                   theme(legend.title = element_blank(),
                         axis.text.x = element_blank(),
                         axis.text.y = element_blank(),
                         axis.ticks.x = element_blank(),
                         axis.ticks.y = element_blank(),
                         axis.title.x = element_blank(),
                         axis.title.y = element_blank()
                        ) + 
                     ggtitle(paste(cnty_name, ",",gsub("-", "_", model_name), ",", emission, sep=" "))
      subplot_name <- paste0(gsub("-", "_", model_name), "_", gsub(" ", "_", cnty_name) , "_")
      assign(x = paste(subplot_name), value = {curr_plot})
      rm(target_annotation)
    }
  }
  # model_namess <- c("bcc_csm1_1_m", "BNU_ESM", "CanESM2", "CNRM_CM5", "GFDL_ESM2G", "GFDL_ESM2M")
  # local_county_names <- unique(local_cnty_fips$st_county)
  # local_county_names <- sapply(local_county_names, function(x) strsplit(x, "_")[[1]], USE.NAMES=FALSE)
  # local_county_names = paste0(tolower(x[2, ]), "_")
  # plot_names <- do.call(paste, expand.grid(model_namess, local_county_names, sep='_', stringsAsFactors=FALSE))
 all_map <- ggarrange(plotlist = list(bcc_csm1_1_m_canyon_ , BNU_ESM_canyon_, CanESM2_canyon_, CNRM_CM5_canyon_, GFDL_ESM2G_canyon_, GFDL_ESM2M_canyon_,
                                      bcc_csm1_1_m_hood_river_,  BNU_ESM_hood_river_,  CanESM2_hood_river_, CNRM_CM5_hood_river_, GFDL_ESM2G_hood_river_, GFDL_ESM2M_hood_river_, 
                                      bcc_csm1_1_m_klickitat_, BNU_ESM_klickitat_, CanESM2_klickitat_, CNRM_CM5_klickitat_,  GFDL_ESM2G_klickitat_, GFDL_ESM2M_klickitat_, 
                                      bcc_csm1_1_m_gilliam_, BNU_ESM_gilliam_, CanESM2_gilliam_, CNRM_CM5_gilliam_, GFDL_ESM2G_gilliam_, GFDL_ESM2M_gilliam_, 
                                      bcc_csm1_1_m_morrow_, BNU_ESM_morrow_, CanESM2_morrow_, CNRM_CM5_morrow_, GFDL_ESM2G_morrow_, GFDL_ESM2M_morrow_,  
                                      bcc_csm1_1_m_umatilla_, BNU_ESM_umatilla_, CanESM2_umatilla_, CNRM_CM5_umatilla_, GFDL_ESM2G_umatilla_, GFDL_ESM2M_umatilla_, 
                                      bcc_csm1_1_m_benton_, BNU_ESM_benton_, CanESM2_benton_, CNRM_CM5_benton_, GFDL_ESM2G_benton_, GFDL_ESM2M_benton_,  
                                      bcc_csm1_1_m_walla_walla_, BNU_ESM_walla_walla_, CanESM2_walla_walla_, CNRM_CM5_walla_walla_, GFDL_ESM2G_walla_walla_, GFDL_ESM2M_walla_walla_,  
                                      bcc_csm1_1_m_franklin_, BNU_ESM_franklin_, CanESM2_franklin_, CNRM_CM5_franklin_, GFDL_ESM2G_franklin_, GFDL_ESM2M_franklin_, 
                                      bcc_csm1_1_m_yakima_, BNU_ESM_yakima_, CanESM2_yakima_, CNRM_CM5_yakima_, GFDL_ESM2G_yakima_, GFDL_ESM2M_yakima_,  
                                      bcc_csm1_1_m_columbia_, BNU_ESM_columbia_, CanESM2_columbia_, CNRM_CM5_columbia_, GFDL_ESM2G_columbia_, GFDL_ESM2M_columbia_, 
                                      bcc_csm1_1_m_grant_,  BNU_ESM_grant_,  CanESM2_grant_, CNRM_CM5_grant_, GFDL_ESM2G_grant_, GFDL_ESM2M_grant_, 
                                      bcc_csm1_1_m_adams_,  BNU_ESM_adams_,  CanESM2_adams_, CNRM_CM5_adams_, GFDL_ESM2G_adams_, GFDL_ESM2M_adams_, 
                                      bcc_csm1_1_m_kittitas_, BNU_ESM_kittitas_, CanESM2_kittitas_, CNRM_CM5_kittitas_, GFDL_ESM2G_kittitas_, GFDL_ESM2M_kittitas_, 
                                      bcc_csm1_1_m_chelan_, BNU_ESM_chelan_, CanESM2_chelan_, CNRM_CM5_chelan_, GFDL_ESM2G_chelan_, GFDL_ESM2M_chelan_,  
                                      bcc_csm1_1_m_douglas_, BNU_ESM_douglas_, CanESM2_douglas_, CNRM_CM5_douglas_, GFDL_ESM2G_douglas_,  GFDL_ESM2M_douglas_, 
                                      bcc_csm1_1_m_okanogan_, BNU_ESM_okanogan_, CanESM2_okanogan_, CNRM_CM5_okanogan_, GFDL_ESM2G_okanogan_, GFDL_ESM2M_okanogan_),
                       ncol = 6, nrow = 17,
                       common.legend = TRUE)
  
  file_save_name <- paste0(emission, "_", time, ".png")
  ggsave(filename = file_save_name, plot = all_map, device = "png",
         width = 20, height = 60, units = "in", dpi=300, path=data_dir, limitsize = FALSE)
  rm(time)
}










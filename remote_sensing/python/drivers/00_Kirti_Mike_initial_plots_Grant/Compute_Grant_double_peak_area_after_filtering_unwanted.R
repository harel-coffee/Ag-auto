##########################################################################################
######
######   Compute Grant double peak area after filtering unwanted plants out
######
######
##########################################################################################
rm(list=ls())
library(data.table)
library(dplyr)

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/01_NDVI_TS/Grant/No_EVI/", 
                   "Grant_10_cloud/Grant_2017/")
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"


grant_2017_double_peak <- read.csv(paste0(data_dir, "double_polygons.csv"), as.is=TRUE)
grant_2017_double_peak <- within(grant_2017_double_peak, remove("geo"))

##########################################################################################
###
###  There is one last extra empty row, drop it
###
##########################################################################################
grant_2017_double_peak <- grant_2017_double_peak[1:(nrow(grant_2017_double_peak)-1), ]
sum(grant_2017_double_peak$ExctAcr)

grant_2017_2Peak_acr_by_cultivar <- grant_2017_double_peak %>% 
                                    group_by(county, year, CropTyp) %>% 
                                    summarise(acreage_by_plant=sum(ExctAcr)) %>% 
                                    data.table()

write.table(grant_2017_2Peak_acr_by_cultivar, 
            file = paste0(data_dir, "grant_2017_all_2Peak_acr_by_cultivar.csv"),
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

double_crop_potential_plants <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), 
                                         as.is=TRUE)

grant_2017_double_peak <- grant_2017_double_peak %>% 
                          filter(CropTyp %in% double_crop_potential_plants$Crop_Type) %>%
                          data.table()

sum(grant_2017_double_peak$ExctAcr) # 4797.035

grant_2017_2Peak_acr_by_cultivar <- grant_2017_double_peak %>% 
                                    group_by(county, year, CropTyp) %>% 
                                    summarise(acreage_by_plant=sum(ExctAcr)) %>% 
                                    data.table()

write.table(grant_2017_2Peak_acr_by_cultivar, 
            file = paste0(data_dir, "grant_2017_potential_2Crop_2Peak_acr_by_cultivar.csv"),
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")




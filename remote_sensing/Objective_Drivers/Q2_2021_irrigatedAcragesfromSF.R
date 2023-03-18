###
### Q3 of objectives.
### In order to compute crop intensity we
### need to have something in denominator:
###
###  acreage of irrigated fields in each county--also county, crop combination.
###  we will do this thing for all combinations of filters: 
###                                       NASS, lastSurveyDate, and perennials 
###
###
############################################################################################s
rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
data_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"

############################################################################################s

double_crop_potens <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"),
                               as.is=TRUE)

years <- c(2018, 2017, 2016)
year <- 2018
NASS <- TRUE
lastSurveyDate <- TRUE
perennials <- TRUE

for (year in years){
  for (NASS in c(TRUE, FALSE)){
    for (lastSurveyDate in c(TRUE, FALSE)){
      for (perennials in c(TRUE, FALSE)){
        ####################
        #################### Read everything here again and again!
        #################### to avoid thinking about it and making mistake!
        ####################

        f_name = paste0("WSDA_DataTable_", year, ".csv")
        WSDA <- read.csv(paste0(data_dir, f_name))

        WSDA$CropTyp <- tolower(WSDA$CropTyp)
        WSDA$DataSrc <- tolower(WSDA$DataSrc)

        East <- pick_eastern_counties(WSDA) # only pick the eastern counties
        #***************************************
        #
        # only pick the irrigated fields
        #
        #***************************************
        East <- filter_out_non_irrigated_datatable(East) 

        if (NASS == TRUE){
          # print (dim(East))
          East <- East %>% filter(DataSrc != "nass")
          nassName = "nassOut"
          # print (dim(East))
        }else{
          nassName = "nassIn"
        }

        if (lastSurveyDate == TRUE){
          # print (dim(East))
          East <- filter_lastSrvyDate_DataTable(East, year)
          # print (dim(East))
          surveyDateName = "correctSurveyDate"
        }else{
          surveyDateName = "wrongSurveyDate"
        }

        if (perennials == TRUE){ 
          # only annuals
          # print (dim(East))
          East <- East %>% filter(CropTyp %in% double_crop_potens$Crop_Type) %>% data.table()
          # print (dim(East))
          perennialsName = "perennialsOut"
        }else{
          perennialsName = "perennialsin"
        }

        acreage_per_county <- East %>% 
                              group_by(county)%>% 
                              summarise(irr_acreage=sum(ExctAcr)) %>%
                              data.table()

        acreage_per_county_corp <- East %>% 
                                   group_by(county, CropTyp)%>% 
                                   summarise(irr_acreage=sum(ExctAcr)) %>%
                                   data.table()

        acreage_per_crop <- East %>% 
                            group_by(CropTyp)%>% 
                            summarise(irr_acreage=sum(ExctAcr)) %>%
                            data.table()
       

        outputName_county = paste("irrigatedAcr_perCounty", nassName, surveyDateName, perennialsName, year, sep = "_")
        outputName_countyCrop = paste("irrigatedAcr_perCountyCrop", nassName, surveyDateName, perennialsName, year, sep = "_")
        outputName_Crop = paste("irrigatedAcr_perCrop", nassName, surveyDateName, perennialsName, year, sep = "_")
        
        output_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/objectives_answer/", 
                             "Q2-Acr_onlyIrrigated_and_other_filters_perCountyandCrop/")
        
        if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)} 


        write.csv(acreage_per_county, 
                  paste0(output_dir, "/percounty/", outputName_county, ".csv"), row.names = F)

        write.csv(acreage_per_county_corp, 
                  paste0(output_dir, "/percountyCrop/", outputName_countyCrop, ".csv"), row.names = F)

        write.csv(acreage_per_crop, 
                  paste0(output_dir, "/perCrop/", outputName_Crop, ".csv"), row.names = F)

      }
    }
  }
}





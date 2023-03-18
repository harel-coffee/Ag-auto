####
#### Q3. Crop intensities in each county, and for each (county, crop) pair, 
#### where NASS, LastSuveyDate, and perennials are filtered out in different combinations.
#### We do this for SOS = 0.3 and SG params = [7, 3]
####

#
# The script first answers Q3.
#

library(data.table)
library(dplyr)
source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)


data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                   "06_list_of_double_cropped_plants/")



denom_dir_base <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                         "objectives_answer/Q2-Acr_onlyIrrigated_and_other_filters_perCountyandCrop/")

denom_dir_percounty <- paste0(denom_dir_base, "percounty/")
denom_dir_percountyCrop <- paste0(denom_dir_base, "percountyCrop/")
denom_dir_perCrop <- paste0(denom_dir_base, "perCrop/")


param_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/")
double_crop_potens <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

output_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/objectives_answer/Q3/"

################################################################################################################
double_SEOS3 <- read.csv(paste0(data_dir, "extended_2cropped_fields_noFilter_SEOS3.csv"))

double_SEOS3 <- pick_eastern_counties(double_SEOS3) # only pick the eastern counties

double_SEOS3 <- filter_out_non_irrigated_datatable(double_SEOS3)

double_SEOS3 <- double_SEOS3 %>%
                filter(SG_params == 73) %>%
                data.table()

double_SEOS3$CropTyp <- tolower(double_SEOS3$CropTyp)
double_SEOS3$DataSrc <- tolower(double_SEOS3$DataSrc)


for (year in c(2016, 2017, 2018)){
  for (NASS in c(TRUE, FALSE)){
    for (lastSurveyDate in c(TRUE, FALSE)){
      for (perennials in c(TRUE, FALSE)){

        curr_2cropped_fields <- double_SEOS3


        if (NASS == TRUE){
          print (dim(curr_2cropped_fields))
          curr_2cropped_fields <- curr_2cropped_fields %>% filter(DataSrc != "nass")
          nassName = "nassOut"
          print (dim(curr_2cropped_fields))
        }else{
          nassName = "nassIn"
        }

        if (lastSurveyDate == TRUE){
          print (dim(curr_2cropped_fields))
          curr_2cropped_fields <- filter_lastSrvyDate_DataTable(curr_2cropped_fields, year)
          print (dim(curr_2cropped_fields))
          surveyDateName = "correctSurveyDate"
        }else{
          surveyDateName = "wrongSurveyDate"
        }

        if (perennials == TRUE){ 
          # only annuals
          print (dim(curr_2cropped_fields))
          curr_2cropped_fields <- curr_2cropped_fields %>% 
                                  filter(CropTyp %in% double_crop_potens$Crop_Type) %>%
                                  data.table()
          print (dim(curr_2cropped_fields))
          perennialsName = "perennialsOut"
        }else{
          perennialsName = "perennialsin"
        }

        #####
        ##### read denominators that are computed previously for Q2.
        #####
        
        a_county = paste("irrigatedAcr_perCounty", nassName, surveyDateName, perennialsName, year, sep = "_")
        a_countyCrop = paste("irrigatedAcr_perCountyCrop", nassName, surveyDateName, perennialsName, year, sep = "_")
        a_Crop = paste("irrigatedAcr_perCrop", nassName, surveyDateName, perennialsName, year, sep = "_")
        

        curr_denominator_perCounty <- read.csv(paste0(denom_dir_percounty, a_county, ".csv"), as.is=TRUE)
        curr_denominator_perCountycrop <- read.csv(paste0(denom_dir_percountyCrop, a_countyCrop, ".csv"), as.is=TRUE)
        curr_denominator_perCrop <- read.csv(paste0(denom_dir_perCrop, a_Crop, ".csv"), as.is=TRUE)

        #####
        ##### compute numerators
        #####
        numerator_per_county <- curr_2cropped_fields %>% 
                                group_by(county)%>% 
                                summarise(acrSum_num=sum(ExctAcr)) %>%
                                data.table()

        numerator_per_county_corp <- curr_2cropped_fields %>% 
                                     group_by(county, CropTyp)%>% 
                                     summarise(acrSum_num=sum(ExctAcr)) %>%
                                     data.table()

        numerator_per_corp <- curr_2cropped_fields %>% 
                              group_by(CropTyp)%>% 
                              summarise(acrSum_num=sum(ExctAcr)) %>%
                              data.table()

        #####
        ##### Do the left join. X should be denominator since it is larger, 
        ##### since, some plants may not be double-cropped and therefore 
        ##### are not present in the numerator.
        #####
        numerator_per_county$county <- as.character(numerator_per_county$county)
        numerator_per_county_corp$county <- as.character(numerator_per_county_corp$county)

        intensity_perCounty <- dplyr::left_join(x=curr_denominator_perCounty, 
                                                y=numerator_per_county,
                                                by=c("county"))

        intensity_perCountyCrop <- dplyr::left_join(x=curr_denominator_perCountycrop, 
                                                    y=numerator_per_county_corp,
                                                    by=c("county", "CropTyp"))
        
        intensity_perCrop <- dplyr::left_join(x=curr_denominator_perCrop, 
                                              y=numerator_per_corp,
                                              by=c("CropTyp"))

        #
        # If a crop is not detected by algorithm as double-cropped
        # there will be NAs in the table. replace them with zero.
        #
        intensity_perCounty$acrSum_num[is.na(intensity_perCounty$acrSum_num)] <- 0
        intensity_perCountyCrop$acrSum_num[is.na(intensity_perCountyCrop$acrSum_num)] <- 0
        intensity_perCrop$acrSum_num[is.na(intensity_perCrop$acrSum_num)] <- 0

        intensity_perCounty$crp_intens <- intensity_perCounty$acrSum_num / intensity_perCounty$acr_sum
        intensity_perCountyCrop$crp_intens <- intensity_perCountyCrop$acrSum_num / intensity_perCountyCrop$acr_sum
        intensity_perCrop$crp_intens <- intensity_perCrop$acrSum_num / intensity_perCrop$acr_sum


        a_county = paste("intens_perCounty", nassName, surveyDateName, perennialsName, year, sep = "_")
        a_countyCrop = paste("intens_perCountyCrop", nassName, surveyDateName, perennialsName, year, sep = "_")
        a_Crop = paste("intens_perCrop", nassName, surveyDateName, perennialsName, year, sep = "_")


        ####
        ####  round the decimals
        ####
        cols <- c("acr_sum", "acrSum_num", "crp_intens")

        intensity_perCounty <- data.table(intensity_perCounty)
        intensity_perCountyCrop <- data.table(intensity_perCountyCrop)
        intensity_perCrop <- data.table(intensity_perCrop)

        intensity_perCounty[,(cols) := round(.SD,2), .SDcols=cols]
        intensity_perCountyCrop[,(cols) := round(.SD,2), .SDcols=cols]
        intensity_perCrop[,(cols) := round(.SD,2), .SDcols=cols]



        write.csv(intensity_perCounty, 
                  paste0(output_dir, "/percounty/", a_county, ".csv"), row.names = F)

        write.csv(intensity_perCountyCrop, 
                  paste0(output_dir, "/percountyCrop/", a_countyCrop, ".csv"), row.names = F)

        write.csv(intensity_perCrop, 
                  paste0(output_dir, "/perCrop/", a_Crop, ".csv"), row.names = F)

        rm(intensity_perCrop, intensity_perCountyCrop, intensity_perCounty)
        rm(numerator_per_county, numerator_per_county_corp, numerator_per_corp)
        rm(curr_denominator_perCounty, curr_denominator_perCountycrop, curr_denominator_perCrop)

      }
    }
  }
}


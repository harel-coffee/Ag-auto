###
### See how much acres we lose if we get rid of fields 
### that are less than 5 acres, and 10 acres. 
###
### Do this only for irrigated fields and for both 
### last survey date and not last survey date. 
### And do this per crop. Leave NASS in. 
###
###  All possible filters: NASS, survey date, irrigation, annuals, anything else?
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

years <- c(2018, 2017)
year = 2018
for (year in years){
  f_name = paste0("WSDA_DataTable_", year, ".csv")
  WSDA <- read.csv(paste0(data_dir, f_name))
  WSDA$CropTyp <- tolower(WSDA$CropTyp)

  East <- pick_eastern_counties(WSDA) # only pick the eastern counties
  East <- East %>%                    # only pick the annual plants
          filter(CropTyp %in% double_crop_potens$Crop_Type ) %>%
          data.table()

  East <- filter_out_non_irrigated_datatable(East) # only pick the irrigated fields

  East_CorrectSurveyDate <- filter_lastSrvyDate_DataTable(East, year)

  ##
  ##   Filter out large fields.
  ##
  East_lessThan5  <- East %>% filter(Acres<=5) %>% data.table()
  East_lessThan10 <- East %>% filter(Acres<=10) %>% data.table()

  East_CorrectSurveyDate_lessThan5  <- East_CorrectSurveyDate %>% filter(Acres<=5) %>% data.table()
  East_CorrectSurveyDate_lessThan10 <- East_CorrectSurveyDate %>% filter(Acres<=10) %>% data.table()

  ##
  ## Compute acreage sum
  ##

  East_Acr_perCrop <- East %>% 
                      group_by(CropTyp)%>% 
                      summarise(acr_sum=sum(Acres)) %>%
                      data.table()
  
  East_CorrectSurveyDate_Acr_perCrop <- East_CorrectSurveyDate %>% 
                                        group_by(CropTyp)%>% 
                                        summarise(acr_sum=sum(Acres)) %>%
                                        data.table()

  East_lessThan5_Acr_perCrop <- East_lessThan5 %>% 
                                group_by(CropTyp)%>% 
                                summarise(acr_sum_lessThan5=sum(Acres)) %>%
                                data.table()
  
  East_lessThan10_Acr_perCrop <- East_lessThan10 %>% 
                                  group_by(CropTyp)%>% 
                                  summarise(acr_sum_lessThan10=sum(Acres)) %>%
                                  data.table()
  
  East_CorrectSurveyDate_lessThan5_Acr_perCrop <- East_CorrectSurveyDate_lessThan5 %>% 
                                                  group_by(CropTyp)%>% 
                                                  summarise(acr_sum_Surv_lessThan5=sum(Acres)) %>%
                                                  data.table()
  
  East_CorrectSurveyDate_lessThan10_Acr_perCrop <- East_CorrectSurveyDate_lessThan10 %>% 
                                                  group_by(CropTyp)%>% 
                                                  summarise(acr_sum_Surv_lessThan10=sum(Acres)) %>%
                                                  data.table()

  Eastern_lossTable <- dplyr::left_join(x = East_Acr_perCrop, 
                                        y = East_lessThan5_Acr_perCrop, 
                                        by = "CropTyp")

  Eastern_lossTable <- dplyr::left_join(x = Eastern_lossTable, 
                                        y = East_lessThan10_Acr_perCrop, 
                                        by = "CropTyp")


  Eastern_SurvCorrect_lossTable <- dplyr::left_join(x = East_CorrectSurveyDate_Acr_perCrop, 
                                                    y = East_CorrectSurveyDate_lessThan5_Acr_perCrop, 
                                                    by = "CropTyp")

  Eastern_SurvCorrect_lossTable <- dplyr::left_join(x = Eastern_SurvCorrect_lossTable, 
                                                    y = East_CorrectSurveyDate_lessThan10_Acr_perCrop, 
                                                    by = "CropTyp")

  
  out_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/AcrLoss_filter_smallFields/"
  write.csv(Eastern_lossTable, 
            paste0(out_dir, "Eastern_lossTable_", year, ".csv"), row.names = F)

  write.csv(Eastern_SurvCorrect_lossTable, 
            paste0(out_dir, "Eastern_SurvCorrect_lossTable_", year, ".csv"), row.names = F)

}





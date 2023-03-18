rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)
options("scipen"=100, "digits"=2)

data_dir <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
                   "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                   "03_02_2012_2018_all_years_not_corrected_last_survey_and_more_columns/")

wsda_not_corrected_years <- readOGR(paste0(data_dir, "WSDACrop_all_crop_irrigation_class.shp"),
                                    layer = "WSDACrop_all_crop_irrigation_class",
                                    GDAL1_integer64_policy = TRUE)

wsda_not_corrected_years <- wsda_not_corrected_years@data
View(sort(unique(wsda_not_corrected_years$CropTyp)))

counties_of_interes <- c("Whitman", "Asotin", "Garfield", "Ferry",
                         "Franklin", "Grant", "Island", "Columbia",
                         "Adams", "Walla Walla", "Benton", "Chelan", "Douglas",
                         "Kittitas", "Klickitat", "Lincoln", "Okanogan",
                         "Pacific", "Pend Oreille", "Spokane", "Stevens", "Whatcom", "Yakima")

wsda_not_corrected_years <- wsda_not_corrected_years %>% 
                            filter(county %in% counties_of_interes) %>% 
                            data.table()

#####
##
##      TOSS NONE and UNKNOWN irrigation_types
##
#####
wsda_not_corrected_years <- wsda_not_corrected_years %>% 
                            filter(!(Irrigtn %in% c("None", "Unknown")))


Acrage <- wsda_not_corrected_years %>% 
          group_by(CropTyp, county, year) %>% 
          summarise(Acrage = sum(ExctAcr))

out_dir <- "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/"
write.csv(Acrage, file = paste0(out_dir, "Acrage_with_all_nonIrrComb.csv"), row.names=FALSE)




none_combination <- c("None/Rill", "None/Sprinkler", "None/Sprinkler/Wheel Line",
                      "None/Wheel Line", "Drip/None", "Center Pivot/None")

wsda_not_corrected_years <- wsda_not_corrected_years %>% 
                            filter(!(Irrigtn %in% none_combination))


Acrage <- wsda_not_corrected_years %>% 
          group_by(CropTyp, county, year) %>% 
          summarise(Acrage = sum(ExctAcr))

write.csv(Acrage, file = paste0(out_dir, "Acrage_without_all_nonIrrComb.csv"), row.names=FALSE)




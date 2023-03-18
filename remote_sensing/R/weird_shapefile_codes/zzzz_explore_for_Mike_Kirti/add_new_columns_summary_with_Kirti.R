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
                   "02_2012_2018_all_years_not_corrected_last_survey/weird_projections/")

########################################################################################################

WSDACrop_all <- readOGR(paste0(data_dir, "WSDACrop_2012_2018_weird_proj_all_years.shp"),
                        layer = "WSDACrop_2012_2018_weird_proj_all_years",
                        GDAL1_integer64_policy = TRUE)


none_combination <- c("None/Rill", "None/Sprinkler", "None/Sprinkler/Wheel Line", 
                      "None/Wheel Line", "Drip/None", "Center Pivot/None")

WSDACrop_all_None_X <- WSDACrop_all[WSDACrop_all@data$Irrigtn %in% none_combination, ]

# Create new column for irrigation
WSDACrop_all@data$Irrication_class <- "irrigated"

WSDACrop_all@data$Irrication_class[WSDACrop_all@data$Irrigtn %in% none_combination ] <- "none_combination"
WSDACrop_all@data$Irrication_class[WSDACrop_all@data$Irrigtn == "None" ] <- "None"
WSDACrop_all@data$Irrication_class[WSDACrop_all@data$Irrigtn == "Unknown" ] <- "Unknown"

hay_types <- c("Alfalfa Hay", "Alfalfa/Grass Hay", "Barley Hay",
               "Clover Hay", "Clover/Grass Hay", "Grass Hay",
               "Hay/Silage , Unknown", "Hay/Silage, Unknown", "Oat Hay",
               "Rye Hay", "Triticale Hay")

WSDACrop_all@data$crop_class <- "Not Hay"
WSDACrop_all@data$crop_class[WSDACrop_all@data$CropTyp %in% hay_types ] <- "Hay except Timothy"
WSDACrop_all@data$crop_class[WSDACrop_all@data$CropTyp == "Timothy" ] <- "Timothy"

write_dir <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
                    "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                    "03_02_2012_2018_all_years_not_corrected_last_survey_and_more_columns/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDACrop_all, 
         dsn = write_dir, 
         layer="WSDACrop_all_crop_irrigation_class", 
         driver="ESRI Shapefile")


Grant_Yakima <- WSDACrop_all[WSDACrop_all@data$county %in% c("Grant", "Yakima"), ]

Grant_Yakima <- data.table(Grant_Yakima@data)
Grant_Yakima_summary <- Grant_Yakima %>%
                         group_by(crop_class, county, year, Irrication_class) %>%
                         summarize(Acre_sum = sum(ExctAcr))


A <- dcast(Grant_Yakima_summary, county + crop_class + Irrication_class  ~ year)
View(A)

Grant_Yakima <- data.table(Grant_Yakima)
Grant_Yakima_Wheats <- Grant_Yakima[Grant_Yakima$CropTyp %in% c("Wheat", "Wheat Fallow")]
Grant_Yakima_Wheats$CropTyp <- as.character(Grant_Yakima_Wheats$CropTyp)
Grant_Yakima_Wheats_summary <- Grant_Yakima_Wheats %>%
                               group_by(CropTyp, county, year, Irrication_class) %>%
                               summarize(Acre_sum = sum(ExctAcr))

Grant_Yakima_Wheats_summary_cast <- dcast(Grant_Yakima_Wheats_summary, county + CropTyp + Irrication_class ~ year)
View(Grant_Yakima_Wheats_summary_cast)







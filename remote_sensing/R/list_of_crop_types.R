###
### List unique crop types
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

crop_types <- data.table()
years <- c(2018, 2017, 2016, 2015)

year <- years[1]
for (year in years){
  f_name = paste0("WSDA_DataTable_", year, ".csv")
  WSDA <- read.csv(paste0(data_dir, f_name))
  WSDA <- subset(WSDA, select=c("CropTyp", "CropGrp"))
  crop_types<-rbind(crop_types, WSDA)
}


crop_types$CropTyp <- tolower(crop_types$CropTyp)
crop_types <- unique(crop_types)
crop_types$CropTyp <- as.character(crop_types$CropTyp)
crop_types <- crop_types[order(CropTyp),]


write.csv(crop_types, 
          file = paste0(param_dir, "CropTypes_and_cropGroups_2ndTry.csv"), 
          row.names=FALSE)
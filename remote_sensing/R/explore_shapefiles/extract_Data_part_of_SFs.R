rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)


##########
########## Directories
##########
data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                   "01_true_shapefiles_separate_years/")

out_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                  "remote_sensing/01_Data_part_not_filtered_2ndTry/")

if (dir.exists(file.path(out_dir)) == F){
  dir.create(path=file.path(out_dir), recursive=T)
}

years <- c("2015", "2016", "2017", "2018")

for (yr in years){
  print (yr)
  WSDA <- readOGR(paste0(data_dir, "WSDACrop_" , yr, "/WSDACrop_", yr, ".shp"),
                  layer = paste0("WSDACrop_", yr) ,
                  GDAL1_integer64_policy = TRUE)
  WSDA <- data.table(WSDA@data)

  write.csv(WSDA, file = paste0(out_dir, "WSDA_DataTable_", yr, ".csv"), row.names=FALSE)

}

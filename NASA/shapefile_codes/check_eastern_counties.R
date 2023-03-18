################################################################################
#####
#####    Oct. 26, 2021
#####
#####    Goal: Check Eastern Counties in WSDA shapefiles (separate years).
#####          
#####          I noticed Pacific county was in one of the eastern SFs.
#####          Want to double check if any eastern county is missing in the files.
#####
##### 
#####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/WSDA_separateYears/")
output_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Body
#

start_time <- Sys.time()

years <- c(2008:2020)

Eastern_cunties <- sort(c("Okanogan", "Chelan", "Kittitas", "Yakima", "Klickitat", 
                          "Douglas", "Benton", "Ferry", "Lincoln", "Adams", "Franklin",
                          "Walla Walla", "Pend Oreille", "Stevens", "Spokane", "Whitman", "Garfield", 
                          "Columbia", "Asotin", "Grant"))
for (yr in years){
  SF <- readOGR(paste0(SF_dir, "WSDA_EW_", yr, ".shp"),
                layer = paste0("WSDA_EW_", yr), 
                GDAL1_integer64_policy = TRUE)

  sf_counties <- sort(unique(SF@data$County))
  
  T = sum(sf_counties %in% Eastern_cunties)
  print (T)

  if (T != 20){
    print (yr)
    print (sf_counties)
  } 
}




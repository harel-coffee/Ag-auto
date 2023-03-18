###
############################################################################################s
rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/", 
                   "0002_final_shapeFiles/000_Eastern_WA/")

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/", 
                    "0002_final_shapeFiles/000_Eastern_WA_CorrectYear/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}
############################################################################################s

years <- c(2017)
year = 2017
for (year in years){
  WSDA <- readOGR(paste0(data_dir, "Eastern_", year, "/Eastern_", year, ".shp"),
                  layer = paste0("Eastern_", year), 
                  GDAL1_integer64_policy = TRUE)

  WSDA_correctYear <- WSDA[grepl(as.character(year), WSDA$LstSrvD), ]

  writeOGR(obj = WSDA_correctYear, 
           dsn = paste0(write_dir, "/", "Eastern_", year, "_correctYear"), 
           layer = paste0("Eastern_", year, "_correctYear"), 
           driver = "ESRI Shapefile")
}





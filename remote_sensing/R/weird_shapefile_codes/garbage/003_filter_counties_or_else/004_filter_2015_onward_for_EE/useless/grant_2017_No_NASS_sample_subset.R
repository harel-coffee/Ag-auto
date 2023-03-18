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
data_dir <- paste0("/Users/hn/Documents/01_research_data", 
                   "/remote_sensing/00_shapeFiles/02_correct_years/", 
                   "05_filtered_shapefiles/Grant2017_NoNass_All_Plants/")

Grant2017_No_Nass <- readOGR(paste0(data_dir, "/Grant2017_No_Nass.shp"),
                             layer = "Grant2017_No_Nass", 
                             GDAL1_integer64_policy = TRUE)

############################################################################
#######
####### 
#######

first_Id = Grant2017_No_Nass@data$ID[1]
samples = Grant2017_No_Nass[Grant2017_No_Nass@data$ID == first_Id, ]

for (plt in unique(Grant2017_No_Nass@data$CropTyp)){
  curr_sp <- Grant2017_No_Nass[Grant2017_No_Nass@data$CropTyp == plt, ]
  n <- min(100, nrow(curr_sp@data))
  curr_sp <- curr_sp[1:n, ]
  samples <- rbind(samples, curr_sp)

}


NASS_dir <- paste0("/Users/hn/Documents/01_research_data", 
                   "/remote_sensing/00_shapeFiles/02_correct_years/", 
                   "05_filtered_shapefiles/Grant2017_NoNass_All_Plants_samples/")
if (dir.exists(file.path(NASS_dir)) == F){
  dir.create(path=file.path(NASS_dir), recursive=T)
}

writeOGR(obj = samples,
         dsn = NASS_dir, 
         layer="Grant2017_No_Nass_samples", 
         driver="ESRI Shapefile")






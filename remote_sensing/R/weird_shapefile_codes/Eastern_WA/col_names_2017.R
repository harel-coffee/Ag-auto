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

# write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
#                     "00_shapeFiles/0002_final_shapeFiles/")
# if (dir.exists(file.path(write_dir)) == F){
#   dir.create(path=file.path(write_dir), recursive=T)
# }

##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/0002_final_shapeFiles/", 
                   "000_Eastern_WA/")

param_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/")
############################################################
############################################################
non_perenials <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

############################################################
years = c("2015", "2016", "2017", "2018")

yr = 2015

WSDACrop <- readOGR(paste0(data_dir, "Eastern_", yr, "/Eastern_", yr, ".shp"),
                  layer = paste0("Eastern_", yr), 
                  GDAL1_integer64_policy = TRUE)

View(sort(colnames(WSDACrop@data)))

WSDAData <- WSDACrop@data
dim(WSDAData)

WSDACrop_Irr <- filter_out_non_irrigated_datatable(WSDAData)
dim(WSDACrop_Irr)

WSDA_nonPerenials <- WSDAData %>%
                     filter(CropTyp %in% non_perenials$Crop_Type)
dim(WSDA_nonPerenials)


Irr_and_nonPer <- WSDACrop_Irr %>%
                  filter(CropTyp %in% non_perenials$Crop_Type)
dim(Irr_and_nonPer)




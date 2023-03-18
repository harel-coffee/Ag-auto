################################################################################
#####
#####    Oct. 29, 2021
#####
#####    Goal: I turned the SFs of WSDA into clean ones; deleted useless columns,
#####          renamed columns so that all SFs have identical col. names. 
#####          However, I kept 5, 6 columns that are still useless, but, who knows what is gonna
#####          comeback and bite me in the ass.
#####
#####          Supriya intersected these new version of WSDA SFs (called 07_intersect_East and 08_intersect_East_Irr)
#####          She computed the area of each field in Sq. miles. 
#####          IDs are missing :( So, if anything is needed to track back, if doable, I do not know how to go back!
#####          
#####          So, now, I want to get rid of these useless columns
#####          Add new ID to each field, convert the area to Acreage.
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
dir_base <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/"
output_dir <- dir_base
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Body
#

start_time <- Sys.time()

################################################
########
########   East 2018
########
SF <- readOGR(paste0(dir_base, "07_intersect_East_2008_2018/", "Intersect_East_cleanCols_2008_2018.shp"),
              layer = "Intersect_East_cleanCols_2008_2018", 
              GDAL1_integer64_policy = TRUE)

SF@data <- subset(SF@data, select=c("Area"))
SF@data <- tibble::rowid_to_column(SF@data, "ID")
SF@data$ID <- as.character(SF@data$ID)
SF@data$ID <- paste0("i", SF@data$ID)

SF@data$Area <- SF@data$Area / 4046.86 # * 640
setnames(SF@data, old=c("Area"), new=c("acreage"))

writeOGR(obj = SF, 
        dsn = paste0(output_dir, "/09_intersect_East_2008_2018_2cols/"), 
        layer = paste0("09_intersect_East_2008_2018_2cols"), 
        driver="ESRI Shapefile")

################################################
########
########   East 2020
########
SF <- readOGR(paste0(dir_base, "07_intersect_East_2008_2020/", "Intersect_East_cleanCols.shp"),
                layer = "Intersect_East_cleanCols", 
                GDAL1_integer64_policy = TRUE)

SF@data <- subset(SF@data, select=c("Area"))
SF@data <- tibble::rowid_to_column(SF@data, "ID")
SF@data$ID <- as.character(SF@data$ID)
SF@data$ID <- paste0("i", SF@data$ID)

SF@data$Area <- SF@data$Area / 4046.86 # * 640
setnames(SF@data, old=c("Area"), new=c("acreage"))

writeOGR(obj = SF, 
        dsn = paste0(output_dir, "/09_intersect_East_2008_2020_2cols/"), 
        layer = paste0("09_intersect_East_2008_2020_2cols"), 
        driver="ESRI Shapefile")

################################################
########
########   East Irrigated - 2018
########

SF <- readOGR(paste0(dir_base, "08_intersect_East_Irr_2008_2018/", "Intersect_East_Irr_cleanCols_2008_2018.shp"),
              layer = "Intersect_East_Irr_cleanCols_2008_2018", 
              GDAL1_integer64_policy = TRUE)

SF@data <- subset(SF@data, select=c("Area", "County"))
SF@data <- tibble::rowid_to_column(SF@data, "ID")
SF@data$ID <- as.character(SF@data$ID)
SF@data$ID <- paste0("i", SF@data$ID)

SF@data$Area <- SF@data$Area / 4046.86 # * 640
setnames(SF@data, old=c("Area"), new=c("acreage"))

Grant <- SF[grepl('Grant', SF$County), ]
SF@data <- subset(SF@data, select=c("ID", "acreage"))

writeOGR(obj = SF, 
        dsn = paste0(output_dir, "/10_intersect_East_Irr_2008_2018_2cols/"), 
        layer = paste0("10_intersect_East_Irr_2008_2018_2cols"), 
        driver="ESRI Shapefile")

################################################################
####
####    Grant - 2018
####
Grant@data <- subset(Grant@data, select=c("ID", "acreage"))
writeOGR(obj = Grant, 
        dsn = paste0(output_dir, "/10_intersect_Grant_Irr_2008_2018_2cols/"), 
        layer = paste0("10_intersect_Grant_Irr_2008_2018_2cols"), 
        driver="ESRI Shapefile")

################################################################
########
########   East Irrigated - 2020
########

SF <- readOGR(paste0(dir_base, "08_intersect_East_Irr_2008_2020/", "Intersect_East_Irr_cleanCols.shp"),
                layer = "Intersect_East_Irr_cleanCols", 
                GDAL1_integer64_policy = TRUE)

SF@data <- subset(SF@data, select=c("Area"))
SF@data <- tibble::rowid_to_column(SF@data, "ID")
SF@data$ID <- as.character(SF@data$ID)
SF@data$ID <- paste0("i", SF@data$ID)

SF@data$Area <- SF@data$Area / 4046.86 # * 640
setnames(SF@data, old=c("Area"), new=c("acreage"))

writeOGR(obj = SF, 
        dsn = paste0(output_dir, "/10_intersect_East_Irr_2008_2020_2cols/"), 
        layer = paste0("10_intersect_East_Irr_2008_2020_2cols"), 
        driver="ESRI Shapefile")

print (Sys.time() - start_time)



var SF = ee.FeatureCollection(intersect_Grant_Irr_2008_2018_2cols_10);
var outfile_name = 'L5_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10_' + wstart_date + "_" + wend_date;


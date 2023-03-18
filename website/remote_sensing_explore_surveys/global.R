# Per_MS Meeting

library(scales)
library(lattice)
library(jsonlite)
library(raster)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)
# library(plotly)
# library(Hmisc)

#################################################################

filter_non_irrigated <- function (dt){
  dt <- dt[!grepl('None', dt$Irrigtn), ] # toss out those with None in irrigation
  dt <- dt[!grepl('Unknown', dt$Irrigtn), ] # toss out Unknown
  return(dt)

}

#################################################################
data_dir <- "/data/hnoorazar/remote_sensing_pre_MS/"


shape_dir_base <- "/data/hnoorazar/remote_sensing_explore_surveys/"

##############################
############################## TRUELY SURVEYED
##############################
WSDA_2017_correct_years_dir <- paste0(shape_dir_base, "WSDACrop_2017_correct_years/")
Grant_2017_correct_years_all_dir <- paste0(shape_dir_base, "Grant_2017_correct_years_all_fields/")
Grant_2017_correct_years_double_dir <- paste0(shape_dir_base, "Grant_2017_correct_years_double_fields/")

Grant_2015_2018_correct_years_all_dir <- paste0(shape_dir_base, 
                                                "/Grant_2015_2018_correct_years_all_fields/")

Grant_2015_2018_correct_years_all_Fs_irrig_dir <- paste0(shape_dir_base, 
                                                         "Grant_2015_2018_correct_years_all_fields_irrigated/")

Grant_2015_2018_correct_years_2_Fs_irrig_dir <- paste0(shape_dir_base, 
                                                       "Grant_2015_2018_correct_years_2_fields_irrigated/")


##############################
############################## before Microsoft
##############################

# shape_dir <- "/data/hnoorazar/remote_sensing_pre_MS/double_2018/"
# mins_file <- rgdal::readOGR(dsn=path.expand(paste0(shape_dir, 
#                                                    "/double_2018.shp")),
#                             layer = "double_2018", 
#                             GDAL1_integer64_policy = TRUE)

# Min_sp <- spTransform(mins_file, CRS("+init=epsg:4326"))

# centroids <- read.csv(paste0(data_dir, "double_crop_centroid.csv"),
#                      as.is=TRUE) %>% data.table()

##############################
##############################
##############################

##############################
##############################    All 2017
##############################

# WSDA_2017_correct_years_SF <- rgdal::readOGR(dsn=path.expand(paste0(WSDA_2017_correct_years_dir, 
#                                                                     "/WSDACrop_2017.shp")),
#                                              layer = "WSDACrop_2017", 
#                                              GDAL1_integer64_policy = TRUE)

# WSDA_2017_correct_years_SF <- spTransform(WSDA_2017_correct_years_SF, CRS("+init=epsg:4326"))

##############################
##############################    Grant all 2017
##############################

# Grant_2017_correct_years_all_SF <- rgdal::readOGR(dsn=path.expand(paste0(Grant_2017_correct_years_all_dir, 
#                                                                     "/Grant_2017.shp")),
#                                              layer = "Grant_2017", 
#                                              GDAL1_integer64_policy = TRUE)

# Grant_2017_correct_years_all_SF <- spTransform(Grant_2017_correct_years_all_SF, CRS("+init=epsg:4326"))

##############################
##############################    Grant double 2017
##############################

# Grant_2017_correct_years_double_SF <- rgdal::readOGR(dsn=path.expand(paste0(Grant_2017_correct_years_double_dir, 
#                                                                     "/Grant_2017.shp")),
#                                              layer = "Grant_2017", 
#                                              GDAL1_integer64_policy = TRUE)


# Grant_2017_correct_years_double_SF <- spTransform(Grant_2017_correct_years_double_SF, CRS("+init=epsg:4326"))


##############################
##############################
##############################
Grant_2015_2018_correct_years_all_SF <- rgdal::readOGR(
                                                    dsn=path.expand(paste0(Grant_2015_2018_correct_years_all_Fs_irrig_dir, 
                                                                           "/Grant_2015_2018.shp")),
                                             layer = "Grant_2015_2018", 
                                             GDAL1_integer64_policy = TRUE)

Grant_2015_2018_correct_years_all_SF <- spTransform(Grant_2015_2018_correct_years_all_SF, CRS("+init=epsg:4326"))

##############################

Grant_2015_2018_correct_years_2_SF <- rgdal::readOGR(
                                                    dsn=path.expand(paste0(Grant_2015_2018_correct_years_2_Fs_irrig_dir, 
                                                                           "/Grant_2015_2018.shp")),
                                             layer = "Grant_2015_2018", 
                                             GDAL1_integer64_policy = TRUE)

Grant_2015_2018_correct_years_2_SF <- spTransform(Grant_2015_2018_correct_years_2_SF, CRS("+init=epsg:4326"))




#######################################################
#######################################################
#######################################################







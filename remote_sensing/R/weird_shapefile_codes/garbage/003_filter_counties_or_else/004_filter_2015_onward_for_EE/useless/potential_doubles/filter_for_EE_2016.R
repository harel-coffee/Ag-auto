rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)
##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"


WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2016/WSDACrop_2016.shp"),
                    layer = "WSDACrop_2016", 
                    GDAL1_integer64_policy = TRUE)
WSDACrop$Notes <- tolower(WSDACrop$Notes)
############################################################################
#######
#######         Counties of interest
#######
coi = c("Okanogan", "Chelan", "Kittitas", "Yakima", "Klickitat",
        "Douglas", "Grant", "Benton", "Ferry", "Lincoln", "Adams",
        "Franklin", "Walla Walla", "Pend Oreille", "Stevens", "Spokane",
        "Whitman", "Garfield", "Columbia",
        "Asotin")
WSDACrop <- WSDACrop[WSDACrop@data$county %in% coi, ]

#
# potential cultivars for double cropping
#
double_crop_potential_plants = read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), as.is=TRUE)

dim(WSDACrop)
WSDACrop <- WSDACrop[WSDACrop@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
dim(WSDACrop)
############################################################################
#######
#######       Batch 1
####### 
############################################################################

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 17784 rows
dim(Whitman)

############################################################################
#######
#######       Batch 2
####### 
############################################################################

Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  # 4285 rows
dim(Grant)

Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 1064 rows
dim(Yakima)

Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 363 rows
dim( Okanogan)

Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 21 rows
dim(Chelan )

Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 0 rows
dim(Kittitas)

Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 1969 rows
dim(Klickitat)

Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 2 rows
dim(Douglas)

############################################################################
#######
#######   Batch 3
####### 
###########################################################################

Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 5 rows
dim(Walla_Walla)

Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 5865 rows
dim(Spokane)

Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 14 rows
dim(Stevens)

Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 7 rows
dim(Garfield)

Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 0 rows
dim(Columbia)

Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 4 rows
dim(Asotin)

Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 3217 rows
dim(Benton)

Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 5470 rows
dim(Adams)

Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 0 rows
dim(Ferry)

Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 1 rows
dim(Pend_Oreille)

Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 12 rows
dim(Lincoln)

Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 1611 rows
dim(Franklin)


batch_A_2016 <- rbind(Whitman)
batch_B_2016 <- rbind(Spokane, Walla_Walla, Grant, Yakima, Okanogan, Chelan, 
                      Kittitas, Douglas, Stevens, Garfield, Columbia,
                      Asotin)
batch_C_2016 <- rbind(Franklin, Benton, Adams, Ferry, Pend_Oreille, Lincoln, Klickitat)

dim(batch_A_2016)
dim(batch_B_2016)
dim(batch_C_2016)
###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_A_2016_dir <- paste0(base_write, "batch_A_2016/")
if (dir.exists(file.path(batch_A_2016_dir)) == F){
  dir.create(path=file.path(batch_A_2016_dir), recursive=T)
}

writeOGR(obj = batch_A_2016, 
         dsn = batch_A_2016_dir, 
         layer="batch_A_2016", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_B_2016_dir <- paste0(base_write, "batch_B_2016/")
if (dir.exists(file.path(batch_B_2016_dir)) == F){
  dir.create(path=file.path(batch_B_2016_dir), recursive=T)
}

writeOGR(obj = batch_B_2016, 
         dsn = batch_B_2016_dir, 
         layer="batch_B_2016", 
         driver="ESRI Shapefile")


batch_C_2016_dir <- paste0(base_write, "batch_C_2016/")
if (dir.exists(file.path(batch_C_2016_dir)) == F){
  dir.create(path=file.path(batch_C_2016_dir), recursive=T)
}

writeOGR(obj = batch_C_2016, 
         dsn = batch_C_2016_dir, 
         layer="batch_C_2016", 
         driver="ESRI Shapefile")

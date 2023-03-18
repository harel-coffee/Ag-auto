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

WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2015/WSDACrop_2015.shp"),
                    layer = "WSDACrop_2015", 
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

WSDACrop <- WSDACrop[WSDACrop@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]

dim(WSDACrop)
############################################################################
#######
#######       Grant
####### 
############################################################################
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  # 4 rows
nrow(Grant)

Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 2130 rows
nrow(Yakima)

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 1 rows
nrow(Whitman)

Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 3 rows
nrow(Okanogan)

Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 0 rows
nrow(Chelan)

Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 224 rows
nrow(Kittitas)

Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 53 rows
nrow(Klickitat)

Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 3037 rows
nrow(Douglas)

############################################################################
#######
####### 
####### 
###########################################################################

Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 5541 rows
nrow(Walla_Walla)

Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 3 rows
nrow(Spokane)

Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 907 rows
nrow(Stevens)

Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 2806 rows
nrow(Garfield)

Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 3336 rows
nrow(Columbia)

Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 875 rows
nrow(Asotin)

###########################################################################
#######
####### 
####### 
###########################################################################

Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] #  215
nrow(Benton)

Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 5 rows
nrow(Adams)

Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 137 rows
nrow(Ferry)

Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 0 rows
nrow(Pend_Oreille)

Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 7349 rows
nrow(Lincoln)

Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 3660 rows
nrow(Franklin)


batch_A_2015 <- rbind(Grant, Yakima, Whitman, Okanogan, Chelan, 
                      Kittitas, Klickitat, Douglas, Lincoln, Franklin, Benton, Adams, Ferry, Pend_Oreille)

batch_B_2015 <- rbind(Walla_Walla, Spokane, Stevens, Garfield, Columbia, Asotin)


###########################################################################
#######
####### 
####### 
###########################################################################

batch_A_2015_dir <- paste0(base_write, "batch_A_2015/")
if (dir.exists(file.path(batch_A_2015_dir)) == F){
  dir.create(path=file.path(batch_A_2015_dir), recursive=T)
}

writeOGR(obj = batch_A_2015, 
         dsn = batch_A_2015_dir, 
         layer="batch_A_2015", 
         driver="ESRI Shapefile")

batch_B_2015_dir <- paste0(base_write, "batch_B_2015/")
if (dir.exists(file.path(batch_B_2015_dir)) == F){
  dir.create(path=file.path(batch_B_2015_dir), recursive=T)
}

writeOGR(obj = batch_B_2015, 
         dsn = batch_B_2015_dir, 
         layer="batch_B_2015", 
         driver="ESRI Shapefile")



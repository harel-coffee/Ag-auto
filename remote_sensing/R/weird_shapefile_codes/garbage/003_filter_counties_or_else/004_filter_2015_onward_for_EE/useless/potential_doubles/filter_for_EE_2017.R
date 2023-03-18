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

WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2017/WSDACrop_2017.shp"),
                    layer = "WSDACrop_2017", 
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

WSDACrop <- WSDACrop[WSDACrop@data$county %in% coi, ]
dim(WSDACrop) # 

WSDACrop <- WSDACrop[WSDACrop@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
dim(WSDACrop) # 
############################################################################
#######
#######     
####### 
############################################################################

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] 
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]
Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ]
Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ]
Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ]
Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 0 rows
Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 1 rows
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 3 rows
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 829 rows
Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 1 rows
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 81  rows
Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 3120 rows
Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 5108 rows
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 751 rows
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 2753 rows

Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 3322 rows
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 80 rows
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 23 rows
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 7687 rows
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 1831 rows

nrow(Grant) # 6576 rows
nrow(Adams) # 81
nrow(Benton) # 1
nrow(Spokane) # 3
nrow(Klickitat) # 1
nrow(Kittitas) # 0
nrow(Chelan) # 0 rows
nrow(Okanogan) # 611 rows
nrow(Franklin) # 1831
nrow(Yakima) # 86 rows


nrow(Columbia) # 3322
nrow(Lincoln) # 7687
nrow(Ferry) # 80
nrow(Pend_Oreille) # 23
nrow(Stevens) # 751

nrow(Whitman) # 4 rows
nrow(Garfield) # 2753
nrow(Walla_Walla) # 5108
nrow(Douglas) # 3120
nrow(Asotin) # 829

############################################################################
#######
#######       Batch 1
####### 
############################################################################

batch_A_2017 <- rbind(Grant, Yakima, Chelan, Kittitas, Klickitat, 
                      Spokane, Benton, Adams,
                      Okanogan, Franklin)

batch_B1_2017 <- rbind(Lincoln)
batch_B2_2017 <- rbind(Columbia, Ferry)  # it seems Ferry and Pend_Oreille do not have problems.
batch_B3_2017 <- rbind(Pend_Oreille, Stevens)

batch_C_2017 <- rbind(Whitman, Garfield, Walla_Walla, Douglas, Asotin)

nrow(batch_A_2017)
nrow(batch_C_2017)

nrow(batch_A_2017) + nrow(batch_B1_2017) + nrow(batch_B2_2017) + nrow(batch_B3_2017) + nrow(batch_C_2017) == nrow(WSDACrop)

###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_A_2017_dir <- paste0(base_write, "batch_A_2017/")
if (dir.exists(file.path(batch_A_2017_dir)) == F){
  dir.create(path=file.path(batch_A_2017_dir), recursive=T)
}

writeOGR(obj = batch_A_2017, 
         dsn = batch_A_2017_dir, 
         layer="batch_A_2017", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_B1_2017_dir <- paste0(base_write, "batch_B1_2017/")
if (dir.exists(file.path(batch_B1_2017_dir)) == F){
  dir.create(path=file.path(batch_B1_2017_dir), recursive=T)
}

writeOGR(obj = batch_B1_2017, 
         dsn = batch_B1_2017_dir, 
         layer="batch_B1_2017", 
         driver="ESRI Shapefile")


batch_B2_2017_dir <- paste0(base_write, "batch_B2_2017/")
if (dir.exists(file.path(batch_B2_2017_dir)) == F){
  dir.create(path=file.path(batch_B2_2017_dir), recursive=T)
}

writeOGR(obj = batch_B2_2017, 
         dsn = batch_B2_2017_dir, 
         layer="batch_B2_2017", 
         driver="ESRI Shapefile")


batch_B3_2017_dir <- paste0(base_write, "batch_B3_2017/")
if (dir.exists(file.path(batch_B3_2017_dir)) == F){
  dir.create(path=file.path(batch_B3_2017_dir), recursive=T)
}

writeOGR(obj = batch_B3_2017, 
         dsn = batch_B3_2017_dir, 
         layer="batch_B3_2017", 
         driver="ESRI Shapefile")

#######
####### 

batch_C_2017_dir <- paste0(base_write, "batch_C_2017/")
if (dir.exists(file.path(batch_C_2017_dir)) == F){
  dir.create(path=file.path(batch_C_2017_dir), recursive=T)
}

writeOGR(obj = batch_C_2017, 
         dsn = batch_C_2017_dir, 
         layer="batch_C_2017", 
         driver="ESRI Shapefile")
#######
####### 

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

WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2018/WSDACrop_2018.shp"),
                    layer = "WSDACrop_2018", 
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

dim(WSDACrop) # 97191
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
Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ]
Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ]
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ]
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] 
Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] 
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ]
Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ]
Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] 
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] 
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ]
Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ]
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] 
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] 
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ]


dim(Whitman) # 18264
dim(Klickitat ) # 1939
dim(Grant) # 4

dim(Asotin) # 0
dim(Garfield) # 0

dim( Okanogan) # 4
dim(Ferry) # 0
dim(Pend_Oreille) # 0
dim(Stevens) # 0

dim(Columbia) # 2
dim(Lincoln) # 1 
dim(Chelan) # 12

dim(Yakima) # 1360
dim(Spokane) # 5689
dim(Franklin) # 2677


dim(Benton) # 2149
dim( Kittitas) # 230
dim(Adams) # 4198
dim(Douglas) # 3240
dim(Walla_Walla) # 2284

############################################################################
#######
#######       Batch 1
####### 
############################################################################

batch_A_2018 <- rbind( Whitman, Klickitat, Grant, Asotin, 
                       Garfield, Okanogan, Ferry, Pend_Oreille, 
                       Stevens, Columbia, Lincoln, Chelan, Kittitas)
batch_B_2018 <- rbind(Yakima, Spokane, Franklin, Benton, Adams, Douglas, Walla_Walla)

dim(batch_A_2018)
dim(batch_B_2018)

dim(batch_A_2018)[1]+dim(batch_B_2018)[1] == dim(WSDACrop)[1]

###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_A_2018_dir <- paste0(base_write, "batch_A_2018/")
if (dir.exists(file.path(batch_A_2018_dir)) == F){
  dir.create(path=file.path(batch_A_2018_dir), recursive=T)
}

writeOGR(obj = batch_A_2018, 
         dsn = batch_A_2018_dir, 
         layer="batch_A_2018", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_B_2018_dir <- paste0(base_write, "batch_B_2018/")
if (dir.exists(file.path(batch_B_2018_dir)) == F){
  dir.create(path=file.path(batch_B_2018_dir), recursive=T)
}

writeOGR(obj = batch_B_2018, 
         dsn = batch_B_2018_dir, 
         layer="batch_B_2018", 
         driver="ESRI Shapefile")

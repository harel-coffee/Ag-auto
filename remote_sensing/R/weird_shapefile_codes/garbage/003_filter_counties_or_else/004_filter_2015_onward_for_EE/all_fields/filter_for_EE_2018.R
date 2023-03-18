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
                     "05_filtered_shapefiles/batches/")


WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2018/WSDACrop_2018.shp"),
                    layer = "WSDACrop_2018", 
                    GDAL1_integer64_policy = TRUE)

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

dim(WSDACrop) # 97191
############################################################################
#######
#######     
####### 
############################################################################

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] 
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  
Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ]
Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] #
Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 
Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ]
Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ]
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] #
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 
Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 
Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ]
Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] 
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] #
Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] #
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] #


dim(Whitman) # 26347
dim(Klickitat ) # 5447
dim( Grant) # 7
dim(Asotin) # 6
dim(Garfield) # 40
dim( Okanogan) # 6
dim(Ferry) # 0
dim(Pend_Oreille) # 0
dim(Stevens) # 2
dim(Columbia) # 15
dim(Lincoln) # 1 
dim(Chelan) # 106

dim(Yakima) # 14368
dim(Spokane) # 10589
dim(Franklin) # 7261


dim(Benton) # 8373
dim( Kittitas) # 3730
dim(Adams) # 6634
dim(Douglas) # 8201
dim(Walla_Walla) # 6058

############################################################################
#######
#######       Batch 1
####### 
############################################################################

batch_1_2018 <- rbind( Whitman, Klickitat, Grant, Asotin, 
                       Garfield, Okanogan, Ferry, Pend_Oreille, 
                       Stevens, Columbia, Lincoln, Chelan )

############################################################################
#######
#######       Batch 2
####### 
############################################################################

batch_2_2018 <- rbind(Yakima, Spokane, Franklin)

############################################################################
#######
#######   Batch 3
####### 
###########################################################################

batch_3_2018 <- rbind(Benton, Kittitas, Adams, Douglas, Walla_Walla)

dim(batch_1_2018)[1]+dim(batch_2_2018)[1]+dim(batch_3_2018)[1] == dim(WSDACrop)[1]

###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_1_2018_dir <- paste0(base_write, "batch_1_2018/")
if (dir.exists(file.path(batch_1_2018_dir)) == F){
  dir.create(path=file.path(batch_1_2018_dir), recursive=T)
}

writeOGR(obj = batch_1_2018, 
         dsn = batch_1_2018_dir, 
         layer="batch_1_2018", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_2_2018_dir <- paste0(base_write, "batch_2_2018/")
if (dir.exists(file.path(batch_2_2018_dir)) == F){
  dir.create(path=file.path(batch_2_2018_dir), recursive=T)
}

writeOGR(obj = batch_2_2018, 
         dsn = batch_2_2018_dir, 
         layer="batch_2_2018", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_3_2018_dir <- paste0(base_write, "batch_3_2018/")
if (dir.exists(file.path(batch_3_2018_dir)) == F){
  dir.create(path=file.path(batch_3_2018_dir), recursive=T)
}

writeOGR(obj = batch_3_2018, 
         dsn = batch_3_2018_dir, 
         layer="batch_3_2018", 
         driver="ESRI Shapefile")

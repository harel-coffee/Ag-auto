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

WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2016/WSDACrop_2016.shp"),
                    layer = "WSDACrop_2016", 
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

############################################################################
#######
#######       Batch 1
####### 
############################################################################

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 25775 rows
dim(Whitman)

batch_1_2016 <- Whitman
############################################################################
#######
#######       Batch 2
####### 
############################################################################

Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  # 8169 rows
dim(Grant)

Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 6334 rows
dim(Yakima)

Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 448 rows
dim( Okanogan)

Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 4457 rows
dim(Chelan )

Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 81 rows
dim(Kittitas)

Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 4876 rows
dim(Klickitat)

Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 6  rows
dim(Douglas)


batch_2_2016 <- rbind(Grant, Yakima, Okanogan, Chelan, Kittitas, Klickitat, Douglas)
############################################################################
#######
#######   Batch 3
####### 
###########################################################################

Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 36 rows
dim(Walla_Walla)

Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 11137 rows
dim(Spokane)

Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 51 rows
dim(Stevens)

Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 10 rows
dim(Garfield)

Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 8 rows
dim(Columbia)

Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 27 rows
dim(Asotin)

Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 6616 rows
dim(Benton)

Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 9120  rows
dim(Adams)

Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 0 rows
dim(Ferry)

Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 13 rows
dim(Pend_Oreille)

Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 34 rows
dim(Lincoln)

Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 3863 rows
dim(Franklin)

batch_3_2016 <- rbind(Walla_Walla, Spokane, Stevens, Garfield, Columbia, 
                      Asotin, Benton, Adams, Ferry, Pend_Oreille, Lincoln, Franklin)

###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_1_2016_dir <- paste0(base_write, "batch_1_2016/")
if (dir.exists(file.path(batch_1_2016_dir)) == F){
  dir.create(path=file.path(batch_1_2016_dir), recursive=T)
}

writeOGR(obj = batch_1_2016, 
         dsn = batch_1_2016_dir, 
         layer="batch_1_2016", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_2_2016_dir <- paste0(base_write, "batch_2_2016/")
if (dir.exists(file.path(batch_2_2016_dir)) == F){
  dir.create(path=file.path(batch_2_2016_dir), recursive=T)
}

writeOGR(obj = batch_2_2016, 
         dsn = batch_2_2016_dir, 
         layer="batch_2_2016", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_3_2016_dir <- paste0(base_write, "batch_3_2016/")
if (dir.exists(file.path(batch_3_2016_dir)) == F){
  dir.create(path=file.path(batch_3_2016_dir), recursive=T)
}

writeOGR(obj = batch_3_2016, 
         dsn = batch_3_2016_dir, 
         layer="batch_3_2016", 
         driver="ESRI Shapefile")

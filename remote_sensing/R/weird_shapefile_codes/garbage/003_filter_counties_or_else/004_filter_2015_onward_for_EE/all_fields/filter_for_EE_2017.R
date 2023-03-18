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


WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2017/WSDACrop_2017.shp"),
                    layer = "WSDACrop_2017", 
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

dim(WSDACrop) # 72140
############################################################################
#######
#######     
####### 
############################################################################

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 60 rows
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  # 17413 rows
Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 590 rows
Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 8116 rows
Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 11 rows
Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 13 rows
Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 40 rows
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 20 rows
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 1724 rows
Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 84 rows
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 158  rows
Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 5382 rows
Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 8364 rows
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 5575 rows
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 3903 rows
Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 4774 rows
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 692 rows
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 958 rows
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 10754 rows
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 3491 rows


dim(Whitman)
dim( Grant)
dim(Yakima)
dim(Chelan)
dim( Kittitas)
dim(Klickitat )
dim(Spokane)
dim(Asotin)
dim(Benton)
dim(Adams)
dim(Garfield)

dim( Okanogan)
dim(Douglas)
dim(Walla_Walla)
dim(Ferry)
dim(Pend_Oreille)


dim(Stevens)
dim(Columbia)
dim(Lincoln)
dim(Franklin)
############################################################################
#######
#######       Batch 1
####### 
############################################################################

batch_1_2017 <- rbind(Whitman, Grant, Yakima, Chelan, Kittitas, Klickitat, Spokane, Asotin, Benton, Adams, Garfield)

############################################################################
#######
#######       Batch 2
####### 
############################################################################

batch_2_2017 <- rbind(Okanogan, Douglas, Walla_Walla, Ferry, Pend_Oreille)

############################################################################
#######
#######   Batch 3
####### 
###########################################################################

batch_3_2017 <- rbind(Stevens, Columbia, Lincoln, Franklin)

dim(batch_1_2017)[1]+dim(batch_2_2017)[1]+dim(batch_3_2017)[1] == dim(WSDACrop)[1]

###########################################################################
#######
####### 
####### 
###########################################################################
#######
####### 
#######
batch_1_2017_dir <- paste0(base_write, "batch_1_2017/")
if (dir.exists(file.path(batch_1_2017_dir)) == F){
  dir.create(path=file.path(batch_1_2017_dir), recursive=T)
}

writeOGR(obj = batch_1_2017, 
         dsn = batch_1_2017_dir, 
         layer="batch_1_2017", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_2_2017_dir <- paste0(base_write, "batch_2_2017/")
if (dir.exists(file.path(batch_2_2017_dir)) == F){
  dir.create(path=file.path(batch_2_2017_dir), recursive=T)
}

writeOGR(obj = batch_2_2017, 
         dsn = batch_2_2017_dir, 
         layer="batch_2_2017", 
         driver="ESRI Shapefile")
#######
####### 
#######
batch_3_2017_dir <- paste0(base_write, "batch_3_2017/")
if (dir.exists(file.path(batch_3_2017_dir)) == F){
  dir.create(path=file.path(batch_3_2017_dir), recursive=T)
}

writeOGR(obj = batch_3_2017, 
         dsn = batch_3_2017_dir, 
         layer="batch_3_2017", 
         driver="ESRI Shapefile")

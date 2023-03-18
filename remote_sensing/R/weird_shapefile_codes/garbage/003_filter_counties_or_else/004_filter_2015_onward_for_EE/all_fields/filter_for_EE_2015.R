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


WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2015/WSDACrop_2015.shp"),
                    layer = "WSDACrop_2015", 
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
#######       Grant
####### 
############################################################################
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]  # 21 rows
nrow(Grant)

Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 15459 rows
nrow(Yakima)

Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 3 rows
nrow(Yakima)

Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 42 rows
nrow(Okanogan)

Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 4 rows
nrow(Chelan)

Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 3708 rows
nrow(Kittitas)

Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 992 rows
nrow(Klickitat)

Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 7642 rows
nrow(Douglas)

batch_1_2015 <- rbind(Grant, Yakima, Whitman, Okanogan, Chelan, Kittitas, Klickitat, Douglas)

############################################################################
#######
####### 
####### 
###########################################################################

Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 10068 rows
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 30 rows
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 4676 rows
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 3850 rows
Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 4756 rows
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 1685 rows

batch_2_2015 <- rbind(Walla_Walla, Spokane, Stevens, Garfield, Columbia, Asotin)



###########################################################################
#######
####### 
####### 
###########################################################################

Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 3941 rows
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 188  rows
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 632 rows
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 0 rows
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 10474 rows
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 8273 rows

batch_3_2015 <- rbind(Benton, Adams, Ferry, Pend_Oreille, Lincoln, Franklin)


###########################################################################
#######
####### 
####### 
###########################################################################

batch_1_2015_dir <- paste0(base_write, "batch_1_2015/")
if (dir.exists(file.path(batch_1_2015_dir)) == F){
  dir.create(path=file.path(batch_1_2015_dir), recursive=T)
}

writeOGR(obj = batch_1_2015, 
         dsn = batch_1_2015_dir, 
         layer="batch_1_2015", 
         driver="ESRI Shapefile")



batch_2_2015_dir <- paste0(base_write, "batch_2_2015/")
if (dir.exists(file.path(batch_2_2015_dir)) == F){
  dir.create(path=file.path(batch_2_2015_dir), recursive=T)
}

writeOGR(obj = batch_2_2015, 
         dsn = batch_2_2015_dir, 
         layer="batch_2_2015", 
         driver="ESRI Shapefile")



batch_3_2015_dir <- paste0(base_write, "batch_3_2015/")
if (dir.exists(file.path(batch_3_2015_dir)) == F){
  dir.create(path=file.path(batch_3_2015_dir), recursive=T)
}

writeOGR(obj = batch_3_2015, 
         dsn = batch_3_2015_dir, 
         layer="batch_3_2015", 
         driver="ESRI Shapefile")


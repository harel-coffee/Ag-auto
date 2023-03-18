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

base_write <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/for_EE/")



WSDACrop <- readOGR(paste0(data_dir, "/WSDACrop_2012_2018_lat_long.shp"),
                     layer = "WSDACrop_2012_2018_lat_long", 
                     GDAL1_integer64_policy = TRUE)

WSDACrop <- WSDACrop[WSDACrop@data$year %in% c(2015, 2016, 2017, 2018), ]

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
Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]

grant_dir <- paste0(base_write, "Grant_2015_2018/")
if (dir.exists(file.path(grant_dir)) == F){
  dir.create(path=file.path(grant_dir), recursive=T)
}

writeOGR(obj = Grant, 
         dsn = grant_dir, 
         layer="Grant_2015_2018", 
         driver="ESRI Shapefile")

rm(Grant, grant_dir)
############################################################################
#######
#######       Yakima
####### 
############################################################################
Yakima <- WSDACrop[grepl('Yakima', WSDACrop$county), ] # 36000 rows

Yakima_dir <- paste0(base_write, "Yakima_2015_2018/")
if (dir.exists(file.path(Yakima_dir)) == F){
  dir.create(path=file.path(Yakima_dir), recursive=T)
}

writeOGR(obj = Yakima, 
         dsn = Yakima_dir, 
         layer="Yakima_2015_2018", 
         driver="ESRI Shapefile")

rm(Yakima, Yakima_dir)

############################################################################
#######
#######       Whitman
####### 
############################################################################
Whitman <- WSDACrop[grepl('Whitman', WSDACrop$county), ] # 52185 rows
Whitman_dir <- paste0(base_write, "Whitman_2015_2018/")
if (dir.exists(file.path(Whitman_dir)) == F){
  dir.create(path=file.path(Whitman_dir), recursive=T)
}

writeOGR(obj = Whitman, 
         dsn = Whitman_dir, 
         layer="Whitman_2015_2018", 
         driver="ESRI Shapefile")

rm(Whitman, Whitman_dir)
############################################################################
#######
#######       Okanogan, Chelan, Kittitas, Klickitat
####### 
###########################################################################

# The following 4 adds up to 32,077
Okanogan <- WSDACrop[grepl('Okanogan', WSDACrop$county), ] # 8612 rows
Chelan <- WSDACrop[grepl('Chelan', WSDACrop$county), ] # 4578 rows
Kittitas <- WSDACrop[grepl('Kittitas', WSDACrop$county), ] # 7532 rows
Klickitat <- WSDACrop[grepl('Klickitat', WSDACrop$county), ] # 11355 rows

Ok_Ch_Ki_Kl <- rbind(Okanogan, Chelan, Kittitas, Klickitat)
rm(Okanogan, Chelan, Kittitas, Klickitat)

Ok_Ch_Ki_Kl_dir <- paste0(base_write, "Ok_Ch_Ki_Kl_2015_2018/")
if (dir.exists(file.path(Ok_Ch_Ki_Kl_dir)) == F){
  dir.create(path=file.path(Ok_Ch_Ki_Kl_dir), recursive=T)
}

writeOGR(obj = Ok_Ch_Ki_Kl, 
         dsn = Ok_Ch_Ki_Kl_dir, 
         layer="Ok_Ch_Ki_Kl_2015_2018", 
         driver="ESRI Shapefile")

rm(Ok_Ch_Ki_Kl, Ok_Ch_Ki_Kl_dir)

############################################################################
#######
#######       Douglas
####### 
###########################################################################

Douglas <- WSDACrop[grepl('Douglas', WSDACrop$county), ] # 21231 rows

Douglas_dir <- paste0(base_write, "Douglas_2015_2018/")
if (dir.exists(file.path(Douglas_dir)) == F){
  dir.create(path=file.path(Douglas_dir), recursive=T)
}

writeOGR(obj = Douglas, 
         dsn = Douglas_dir, 
         layer="Douglas_2015_2018", 
         driver="ESRI Shapefile")

rm(Douglas, Douglas_dir)

############################################################################
#######
#######       Benton, Adams, Ferry, Pend_Oreille
####### 
###########################################################################

# The following 4 adds up to 37436
Benton <- WSDACrop[grepl('Benton', WSDACrop$county), ] # 19014 rows
Adams <- WSDACrop[grepl('Adams', WSDACrop$county), ] # 16100  rows
Ferry <- WSDACrop[grepl('Ferry', WSDACrop$county), ] # 1324 rows
Pend_Oreille <- WSDACrop[grepl('Pend Oreille', WSDACrop$county), ] # 998 rows

Bent_Adams_Ferry_Pend <- rbind(Benton, Adams, Ferry, Pend_Oreille)
rm(Benton, Adams, Ferry, Pend_Oreille)

Bent_Adams_Ferry_Pend_dir <- paste0(base_write, "Bent_Adams_Ferry_Pend_2015_2018/")
if (dir.exists(file.path(Bent_Adams_Ferry_Pend_dir)) == F){
  dir.create(path=file.path(Bent_Adams_Ferry_Pend_dir), recursive=T)
}

writeOGR(obj = Bent_Adams_Ferry_Pend, 
         dsn = Bent_Adams_Ferry_Pend_dir, 
         layer="Bent_Adams_Ferry_Pend_2015_2018", 
         driver="ESRI Shapefile")

rm(Bent_Adams_Ferry_Pend, Bent_Adams_Ferry_Pend_dir)

############################################################################
#######
#######       Lincoln, Franklin
####### 
###########################################################################
Lincoln <- WSDACrop[grepl('Lincoln', WSDACrop$county), ] # 21254 rows
Franklin <- WSDACrop[grepl('Franklin', WSDACrop$county), ] # 22888 rows

Lincoln_Franklin <- rbind(Lincoln, Franklin)
rm(Lincoln, Franklin)

Lincoln_Franklin_dir <- paste0(base_write, "Lincoln_Franklin_2015_2018/")
if (dir.exists(file.path(Lincoln_Franklin_dir)) == F){
  dir.create(path=file.path(Lincoln_Franklin_dir), recursive=T)
}

writeOGR(obj = Lincoln_Franklin, 
         dsn = Lincoln_Franklin_dir, 
         layer="Lincoln_Franklin_2015_2018", 
         driver="ESRI Shapefile")

rm(Lincoln_Franklin, Lincoln_Franklin_dir)

############################################################################
#######
#######       Walla_Walla, Spokane
####### 
###########################################################################

Walla_Walla <- WSDACrop[grepl('Walla Walla', WSDACrop$county), ] # 24526 rows
Spokane <- WSDACrop[grepl('Spokane', WSDACrop$county), ] # 21776 rows

Walla_Spokane <- rbind(Walla_Walla, Spokane)
rm(Walla_Walla, Spokane)

Walla_Spokane_dir <- paste0(base_write, "Walla_Spokane_2015_2018/")
if (dir.exists(file.path(Walla_Spokane_dir)) == F){
  dir.create(path=file.path(Walla_Spokane_dir), recursive=T)
}

writeOGR(obj = Walla_Spokane, 
         dsn = Walla_Spokane_dir, 
         layer="Walla_Spokane_2015_2018", 
         driver="ESRI Shapefile")

rm(Walla_Spokane, Walla_Spokane_dir)


############################################################################
#######
#######       Stevens, Garfield, Columbia, Asotin
####### 
###########################################################################

# The following 4 adds up to 31,102
Stevens <- WSDACrop[grepl('Stevens', WSDACrop$county), ] # 10304 rows
Garfield <- WSDACrop[grepl('Garfield', WSDACrop$county), ] # 7803 rows
Columbia <- WSDACrop[grepl('Columbia', WSDACrop$county), ] # 9553  rows
Asotin <- WSDACrop[grepl('Asotin', WSDACrop$county), ] # 3442 rows

Stv_Gar_Col_Aso <- rbind(Stevens, Garfield, Columbia, Asotin)
rm(Stevens, Garfield, Columbia, Asotin)

Stv_Gar_Col_Aso_dir <- paste0(base_write, "Stv_Gar_Col_Aso_2015_2018/")
if (dir.exists(file.path(Stv_Gar_Col_Aso_dir)) == F){
  dir.create(path=file.path(Stv_Gar_Col_Aso_dir), recursive=T)
}

writeOGR(obj = Stv_Gar_Col_Aso, 
         dsn = Stv_Gar_Col_Aso_dir, 
         layer="Stv_Gar_Col_Aso_2015_2018", 
         driver="ESRI Shapefile")

rm(Stv_Gar_Col_Aso, Stv_Gar_Col_Aso_dir)





rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)



SF_dir_base <- paste0("/Users/hn/Documents/01_research_data/", 
                      "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")


train_dir <- "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"


test <- data.table(read.csv(paste0(train_dir, "test20_split_2Bconsistent_Oct17.csv")))
train <-  data.table(read.csv(paste0(train_dir, "train80_split_2Bconsistent_Oct17.csv")))
GT = rbind(train, test)
################################################################################
yr = 2017
in_dir <- paste0(SF_dir_base, "Eastern_", yr, "/")
Eastern_2017 <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                        layer = paste0("Eastern_", yr), 
                        GDAL1_integer64_policy = TRUE)

Grant_2017 <- Eastern_2017[grepl("Grant", Eastern_2017$county), ]
Grant_2017 <- Grant_2017[Grant_2017@data$ID %in% GT$ID, ]




yr = 2018
in_dir <- paste0(SF_dir_base, "Eastern_", yr, "/")
Eastern <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                        layer = paste0("Eastern_", yr), 
                        GDAL1_integer64_policy = TRUE)

F <- Eastern[grepl("Franklin", Eastern$county), ]
F <- F[F@data$ID %in% GT$ID, ]

Y <- Eastern[grepl("Yakima", Eastern$county), ]
Y <- Y[Y@data$ID %in% GT$ID, ]




yr = 2016
in_dir <- paste0(SF_dir_base, "Eastern_", yr, "/")
Eastern <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                        layer = paste0("Eastern_", yr), 
                        GDAL1_integer64_policy = TRUE)

Adams <- Eastern[grepl("Adams", Eastern$county), ]
Adams <- Adams[Adams@data$ID %in% GT$ID, ]


Benton <- Eastern[grepl("Benton", Eastern$county), ]
Benton <- Benton[Benton@data$ID %in% GT$ID, ]


yr = 2015
in_dir <- paste0(SF_dir_base, "Eastern_", yr, "/")
Eastern <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                        layer = paste0("Eastern_", yr), 
                        GDAL1_integer64_policy = TRUE)

Walla <- Eastern[grepl("Walla Walla", Eastern$county), ]
Walla <- Walla[Walla@data$ID %in% GT$ID, ]
dim(Walla)

train_SF = rbind(Grant_2017, F, Y, Adams, Walla, Benton)

train_ = train_SF[train_SF@data$ID %in% train$ID, ]
test_ = train_SF[train_SF@data$ID %in% test$ID, ]


train_@data$TL = "train"
test_@data$TL = "test"

train_SF = rbind(train_, test_)

write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                      "00_shapeFiles/")

writeOGR(obj = train_SF, 
         dsn = paste0(write_dir, "/NASA_GT_set/"), 
         layer = "NASA_GT_set", 
         driver="ESRI Shapefile")

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


data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                   "01_true_shapefiles_separate_years/")


years = seq(2015, 2018)
yr = 2017

for (yr in years){
  WSDA <- readOGR(paste0(data_dir, "WSDACrop_", yr, "/WSDACrop_", yr, ".shp"),
                  layer = paste0("WSDACrop_", yr), 
                  GDAL1_integer64_policy = TRUE)
  WSDA <- data.table(WSDA@data)

  WSDA <- subset(WSDA, select=c("ID", "Irrigtn"))
  WSDA$Irrigtn <- tolower(WSDA$Irrigtn)

  Irrigtn_NA <- dplyr::filter(WSDA, is.na(Irrigtn))
  Irrigtn_NA <- WSDA %>%
                filter(is.na(Irrigtn))
  
  # this is not working well. It drops every rows with NA in it
  Irrigtn_nonNA <- na.omit(WSDA, cols = c("Irrigtn"))

  Irrigtn_None <- WSDA %>% filter(Irrigtn == "none")
  Irrigtn_NoneCombination <- WSDA[grepl('none', WSDA$Irrigtn), ]

  dim(WSDA)
  dim(Irrigtn_None)
  dim(Irrigtn_NoneCombination)
  dim(Irrigtn_NA)
  dim(Irrigtn_nonNA)

  dt <- WSDA
  dt$Irrigtn[is.na(dt$Irrigtn)]


}




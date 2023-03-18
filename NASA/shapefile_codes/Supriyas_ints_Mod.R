#######################################################
#####
#####    Oct. 26, 2021
#####
#####    Goal: Explore, Clean, and compute Area of each polygon
#####          in 2020_I_Mod SF. This is the SF that Supriya created
#####          for Eastern WA.
#####          
#####
##### 
#####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)


##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/2020_I_Mod/")

output_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")

if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Body
#

start_time <- Sys.time()

SF <- readOGR(paste0(SF_dir, "2020_I_Mod.shp"),
                     layer = "2020_I_Mod", 
                     GDAL1_integer64_policy = TRUE)

print (Sys.time() - start_time)
#
# Check irrigation stuff
#


#######################################################
#####
#####    Aug. 17, 2022
#####
#####    Goal: 
#####        Write Statistics of the SFs in the paper; crop mix, # of fields, #irrigated acres.
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



##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/Eastern_2017/")
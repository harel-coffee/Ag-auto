################################################################################
#####
#####    Jan. 12, 2022
#####    Goal: Satisfying everyone in a dysfunctional "group"!!!
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
dir_base <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/"

param_dir <- "/Users/hn/Documents/01_research_data/NASA/parameters/"
output_dir <- dir_base
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Body
#

start_time <- Sys.time()

################################################
########
########
########
evaluation_set <- data.table(read.csv(paste0(param_dir, "evaluation_set.csv")))

A <- copy(evaluation_set)
A <-  A[, .(mean_acr = mean(Acres)), by = c("CropTyp")]

A <- copy(evaluation_set)
A <-  A[, .(median_acr = median(Acres)), by = c("CropTyp")]


A <- copy(evaluation_set)
A <-  A[, .(mean_acr = mean(Acres), sd_acr = sd(Acres)), by = c("CropTyp")]


A <- copy(evaluation_set)
A <- A %>% group_by(CropTyp) %>% summarise(count = n_distinct(ID)) %>% data.table()

A <- copy(evaluation_set)
A <-  A %>% filter(Acres<= 5) %>% data.table()
A <- A %>% group_by(CropTyp) %>% summarise(count = n_distinct(ID)) %>% data.table()


A <- copy(evaluation_set)
A <-  A %>% filter(Acres>5) %>% data.table()
A <-  A %>% filter(Acres<=10) %>% data.table()
A <- A %>% group_by(CropTyp) %>% summarise(count = n_distinct(ID)) %>% data.table()




A <- copy(evaluation_set)
A <-  A %>% filter(Acres<=3) %>% data.table()
A <- A %>% group_by(CropTyp) %>% summarise(count = n_distinct(ID)) %>% data.table()










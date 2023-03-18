####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
####

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

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/")

##############################################################################

SF_year = 2018
given_county = "Grant"

# WSDA <- readOGR(paste0(SF_dir, "Eastern_", SF_year, "/Eastern_", SF_year, ".shp"),
#                        layer = paste0("Eastern_", SF_year), 
#                        GDAL1_integer64_policy = TRUE)
# WSDA <- WSDA@data

WSDA <- read.csv(paste0(data_dir, "WSDA_DataTable_", SF_year, ".csv"))
WSDA$CropType <- tolower(WSDA$CropTyp)

# Eastern <- pick_eastern_counties(WSDA)

Grant <- WSDA %>% 
         filter(county == "Grant") %>% 
         data.table()

dim(Grant) # 17705 for 2017                   1
sum(Grant$ExctAcr) # 794316.67 for 2017       2

Grant_surveyedGivenYr <- Grant[grepl(as.character(SF_year), Grant$LstSrvD), ]
dim(Grant_surveyedGivenYr) # 17413 for 2017                    3
sum(Grant_surveyedGivenYr$ExctAcr) # 790386.06 for 2017        4

# Grant_double_by_notes <- filter_double_by_Notes(Grant_2017)
# dim(Grant_double_by_notes)
# sum(Grant_double_by_notes$ExctAcr)


# Grant_double_by_notes_surveyedGivenYr <- Grant_double_by_notes[grepl(as.character(SF_year), Grant_double_by_notes$LstSrvD), ]
# dim(Grant_double_by_notes_surveyedGivenYr)
# sum(Grant_double_by_notes_surveyedGivenYr$ExctAcr)

########
######## Filter out non-irrigated
########

# for sanity again
Grant <- WSDA %>% 
         filter(county == "Grant") %>% 
         data.table()
dim(Grant)               # 5

Grant_Irrigated <- filter_out_non_irrigated_datatable(Grant)
dim(Grant_Irrigated) # 13793 for 2017                      6
sum(Grant_Irrigated$ExctAcr) # 521766.7 for 2017           7

Grant_Irrigated_surveyedGivenYr <- Grant_Irrigated[grepl(as.character(SF_year), Grant_Irrigated$LstSrvD), ]
dim(Grant_Irrigated_surveyedGivenYr) # 13760 for 2017                   8
sum(Grant_Irrigated_surveyedGivenYr$ExctAcr) # 521396.355 for 2017      9

Grant_Irrigated_doubleNotes <- filter_double_by_Notes(Grant_Irrigated)
dim(Grant_Irrigated_doubleNotes)                         # 10
sum(Grant_Irrigated_doubleNotes$ExctAcr)                 # 11

GrantIrrigated_surveyedGivenYr_doubleNotes <- filter_double_by_Notes(Grant_Irrigated_surveyedGivenYr)
dim(GrantIrrigated_surveyedGivenYr_doubleNotes)          # 12
sum(GrantIrrigated_surveyedGivenYr_doubleNotes$ExctAcr)  # 13

########
######## Perennials subsection
########
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
double_crop_pots <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

Grant_Irrigated_poten <- Grant_Irrigated %>% 
                         filter(CropTyp %in% double_crop_pots$Crop_Type)

dim(Grant_Irrigated_poten) # 4575 for 2017                      # 14
sum(Grant_Irrigated_poten$ExctAcr) # 256314.4 for 2017          # 15

Grant_Irrigated_poten_surveyedGivenYr <- Grant_Irrigated_poten[grepl(as.character(SF_year), Grant_Irrigated_poten$LstSrvD), ]
dim(Grant_Irrigated_poten_surveyedGivenYr) # 4575      # 16

Grant_Irrigated_poten_surveyedGivenYr_double <- filter_double_by_Notes(Grant_Irrigated_poten_surveyedGivenYr)
dim(Grant_Irrigated_poten_surveyedGivenYr_double) # 241                   # 17
sum(Grant_Irrigated_poten_surveyedGivenYr_double$ExctAcr) # 16411.14      # 18

########
########   Toss out NASS and repeat everything again
########
Grant <- WSDA %>% 
         filter(county == "Grant") %>% 
         data.table()

Grant <- Grant %>%
         filter( DataSource != "NASS") %>%
         data.table()

dim(Grant) # 9781
sum(Grant$ExctAcr) # 309449.6

Grant_surveyedGivenYr <- Grant[grepl(as.character(SF_year), Grant$LstSrvD), ]
dim(Grant_surveyedGivenYr) # 9514
sum(Grant_surveyedGivenYr$ExctAcr) # 305797.06


Grant_Irrigated <- filter_out_non_irrigated_datatable(Grant)
dim(Grant_Irrigated) # 9200
sum(Grant_Irrigated$ExctAcr) # 296994.6



Grant_Irrigated_surveyedGivenYr <- Grant_Irrigated[grepl(as.character(SF_year), Grant_Irrigated$LstSrvD), ]
dim(Grant_Irrigated_surveyedGivenYr) # 9175
sum(Grant_Irrigated_surveyedGivenYr$ExctAcr) # 296713.7


Grant_Irrigated_doubleNotes <- filter_double_by_Notes(Gran_Irrigated)
dim(Grant_Irrigated_doubleNotes) # 115
sum(Grant_Irrigated_doubleNotes$ExctAcr) # 9370.81388


Grant_Irrigated_surveyedGivenYr_doubleNotes <- filter_double_by_Notes(Grant_Irrigated_surveyedGivenYr)
dim(Grant_Irrigated_surveyedGivenYr_doubleNotes)
sum(Grant_Irrigated_surveyedGivenYr_doubleNotes$ExctAcr)


Grant_Irrigated_poten <- Grant_Irrigated %>% 
                         filter(CropType %in% double_crop_pots$Crop_Type)

dim(Grant_Irrigated_poten) # 2248
sum(Grant_Irrigated_poten$ExctAcr) # 


Grant_Irrigated_poten_surveyedGivenYr <- Grant_Irrigated_poten[grepl(as.character(SF_year), Grant_Irrigated_poten$LstSrvD), ]
dim(Grant_Irrigated_poten_surveyedGivenYr) # 2248


Grant_Irrigated_poten_surveyedGivenYr_double <- filter_double_by_Notes(Grant_Irrigated_poten_surveyedGivenYr)
dim(Grant_Irrigated_poten_surveyedGivenYr_double) # 76
sum(Grant_Irrigated_poten_surveyedGivenYr_double$ExctAcr) # 7028.4




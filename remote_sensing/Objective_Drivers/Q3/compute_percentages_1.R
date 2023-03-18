####
#### Q1. For each crop what percentage shows as double-cropped
####

#### and then

####
#### Q2. Of all double-cropped lands, what is each crops share?
####

#
# The script first answers Q2. and then computes Q1.
#

library(data.table)
library(dplyr)
source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)

 
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                   "06_list_of_double_cropped_plants/")

param_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/")

potential_plants <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

################################################################################################################
double_SEOS3 <- read.csv(paste0(data_dir, "extended_double_cropped_plants_SEOS3.csv"))
extended_double_cropped_plants_SEOS3 <- double_SEOS3

double_SEOS3 <- filter_out_non_irrigated_datatable(double_SEOS3)


double_SEOS3 <- double_SEOS3 %>%
                group_by(county, image_year, SG_params, CropTyp) %>%
                summarize(acr_sum = sum(Acres, na.rm = TRUE)) %>%
                data.table()

double_SEOS3 <- double_SEOS3 %>%
                filter(CropTyp %in% potential_plants$Crop_Type) %>%
                data.table()

double_SEOS3_73 <- double_SEOS3 %>%
                   filter(SG_params == 73) %>%
                   data.table()

double_SEOS3_73 <- setorderv(double_SEOS3_73, c("image_year", "acr_sum"), c(1, -1))

double_SEOS3_73_DoubleCroppedArea_perCounty <- double_SEOS3_73 %>%
                                               group_by(county, image_year, SG_params) %>%
                                               summarize(countys_double_area = sum(acr_sum, na.rm = TRUE)) %>%
                                               data.table()

double_SEOS3_73 <- dplyr::left_join(x=double_SEOS3_73, 
                                    y=double_SEOS3_73_DoubleCroppedArea_perCounty)


double_SEOS3_73_summ_w_CropSharesPerc <- double_SEOS3_73
double_SEOS3_73_summ_w_CropSharesPerc$cropTypeShare <- (double_SEOS3_73_summ_w_CropSharesPerc$acr_sum / double_SEOS3_73_summ_w_CropSharesPerc$countys_double_area)*100
double_SEOS3_73_summ_w_CropSharesPerc <- data.table(double_SEOS3_73_summ_w_CropSharesPerc)

cols <- c("cropTypeShare")
double_SEOS3_73_summ_w_CropSharesPerc[,(cols) := round(.SD,2), .SDcols=cols]


answer_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                     "objectives_answer/list_of_doubleCropped_fields/")
write.csv(double_SEOS3_73_summ_w_CropSharesPerc, 
          file = paste0(answer_dir, "double_SEOS3_73_summary_with_CropSharesPerc.csv"))

#######
####### Read the shapefiles, filter out NASS, last survey date, irrigated, perenials and compute area per crop.
#######
shapefile_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"

shapefile_2016 <- read.csv(paste0(shapefile_dir, "WSDA_DataTable_2016.csv"), as.is=TRUE)

####
#### Filter out garbage
####
shapefile_2016 <- shapefile_2016 %>%
                  filter(DataSrc != "NASS") %>%
                  data.table()

shapefile_2016 <- shapefile_2016[grepl(as.character(2016), shapefile_2016$LstSrvD), ]
shapefile_2016 <- filter_out_non_irrigated_datatable(shapefile_2016)

shapefile_2016$CropTyp <- tolower(shapefile_2016$CropTyp)
shapefile_2016 <- shapefile_2016 %>%
                  filter(CropTyp %in% potential_plants$Crop_Type) %>%
                  data.table()

#####
############  2017
#####
shapefile_2017 <- read.csv(paste0(shapefile_dir, "WSDA_DataTable_2017.csv"), as.is=TRUE)

####
#### Filter out garbage
####
shapefile_2017 <- shapefile_2017 %>%
                  filter(DataSrc != "NASS") %>%
                  data.table()

shapefile_2017 <- shapefile_2017[grepl(as.character(2017), shapefile_2017$LstSrvD), ]
shapefile_2017 <- filter_out_non_irrigated_datatable(shapefile_2017)

shapefile_2017$CropTyp <- tolower(shapefile_2017$CropTyp)
shapefile_2017 <- shapefile_2017 %>%
                  filter(CropTyp %in% potential_plants$Crop_Type) %>%
                  data.table()

#####
############  2018
#####

shapefile_2018 <- read.csv(paste0(shapefile_dir, "WSDA_DataTable_2018.csv"), as.is=TRUE)

####
#### Filter out garbage
####
shapefile_2018 <- shapefile_2018 %>%
                  filter(DataSrc != "NASS") %>%
                  data.table()

shapefile_2018 <- shapefile_2018[grepl(as.character(2018), shapefile_2018$LstSrvD), ]
shapefile_2018 <- filter_out_non_irrigated_datatable(shapefile_2018)

shapefile_2018$CropTyp <- tolower(shapefile_2018$CropTyp)
shapefile_2018 <- shapefile_2018 %>%
                  filter(CropTyp %in% potential_plants$Crop_Type) %>%
                  data.table()


############
needed_columns <- c("CropTyp", "Acres", "Irrigtn", "LstSrvD", 
                    "DataSrc", "county", "ExctAcr")

shapefile_2016 <- subset(shapefile_2016, select=needed_columns)
shapefile_2017 <- subset(shapefile_2017, select=needed_columns)
shapefile_2018 <- subset(shapefile_2018, select=needed_columns)

shapefile_2016$shapeYr <- 2016
shapefile_2017$shapeYr <- 2017
shapefile_2018$shapeYr <- 2018


threeYears_shapefile <- rbind(shapefile_2016, shapefile_2017, shapefile_2018)

threeYears_shapefile <- threeYears_shapefile %>%
                        group_by(shapeYr, county, CropTyp) %>%
                        summarize(acr_sum_perCrop = sum(Acres, na.rm = TRUE)) %>%
                        data.table()

double_SEOS3_73 <- within(double_SEOS3_73, remove(countys_double_area))
setnames(double_SEOS3_73, old=c("image_year"), new=c("shapeYr"))


double_SEOS3_73 <- setorderv(double_SEOS3_73, c("shapeYr", "county", "CropTyp"), c(1, 1, 1))
threeYears_shapefile <- setorderv(threeYears_shapefile, c("shapeYr", "county", "CropTyp"), c(1, 1, 1))


A <- dplyr::left_join(x = double_SEOS3_73, 
                      y = threeYears_shapefile)


A <- setorderv(A, c("shapeYr", "county", "acr_sum"), c(1, 1, -1))

A$CropPecen_doubleCropped <- 100 * (A$acr_sum / A$acr_sum_perCrop)

A <- data.table(A)
cols <- c("CropPecen_doubleCropped")
A[,(cols) := round(.SD,2), .SDcols=cols]


answer_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                     "objectives_answer/list_of_doubleCropped_fields/")
write.csv(A, 
          file = paste0(answer_dir, "double_SEOS3_73_summary_w_Q1.csv"))



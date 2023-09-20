
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


SF_dir = "/Users/hn/Documents/01_research_data/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/"
pred_dir = "/Users/hn/Documents/01_research_data/NASA/RegionalStatData/"

all_preds = data.table(read.csv(paste0(pred_dir, "all_preds_overSample.csv")))

all_preds <- within(all_preds, remove("CropTyp", "Acres", "ExctAcr", "Irrigtn", "LstSrvD", "DataSrc"))

WSDA_2015 <- readOGR(paste0(SF_dir, "Eastern_2015", "/", "Eastern_2015.shp"),
                            layer = "Eastern_2015", 
                            GDAL1_integer64_policy = TRUE)


WSDA_2016 <- readOGR(paste0(SF_dir, "Eastern_2016", "/", "Eastern_2016.shp"),
                            layer = "Eastern_2016", 
                            GDAL1_integer64_policy = TRUE)


WSDA_2017 <- readOGR(paste0(SF_dir, "Eastern_2017", "/", "Eastern_2017.shp"),
                            layer = "Eastern_2017", 
                            GDAL1_integer64_policy = TRUE)


WSDA_2018 <- readOGR(paste0(SF_dir, "Eastern_2018", "/", "Eastern_2018.shp"),
                            layer = "Eastern_2018", 
                            GDAL1_integer64_policy = TRUE)


WallaWalla_2015 = WSDA_2015[WSDA_2015@data$county == "Walla Walla", ]
AdamsBenton_2016 = WSDA_2016[WSDA_2016@data$county %in% c("Adams", "Benton"), ]
Grant_2017 = WSDA_2017[WSDA_2017@data$county == "Grant", ]
FranklinYakima_2018 = WSDA_2018[WSDA_2018@data$county %in% c("Franklin", "Yakima"), ]


WallaWalla_preds = all_preds[all_preds$county == "Walla Walla"]
AdamsBenton_preds = all_preds[all_preds$county %in% c("Adams", "Benton")]
Grant_preds = all_preds[all_preds$county == "Grant"]
FranklinYakima_preds = all_preds[all_preds$county %in% c("Franklin", "Yakima")]


WallaWalla_2015 = WallaWalla_2015[WallaWalla_2015@data$ID %in% WallaWalla_preds$ID, ]
AdamsBenton_2016 = AdamsBenton_2016[AdamsBenton_2016@data$ID %in% AdamsBenton_preds$ID, ]
Grant_2017 = Grant_2017[Grant_2017@data$ID %in% Grant_preds$ID, ]
FranklinYakima_2018 = FranklinYakima_2018[FranklinYakima_2018@data$ID %in% FranklinYakima_preds$ID, ]


all_preds <- within(all_preds, remove("county"))

dim(WallaWalla_2015)
WallaWalla_2015@data = dplyr::left_join(x = WallaWalla_2015@data, 
                                        y = all_preds, 
                                        by = "ID")

dim(WallaWalla_2015)

AdamsBenton_2016@data = dplyr::left_join(x = AdamsBenton_2016@data, 
                                         y = all_preds, 
                                         by = "ID")


Grant_2017@data = dplyr::left_join(x = Grant_2017@data, 
                                   y = all_preds, 
                                   by = "ID")

FranklinYakima_2018@data = dplyr::left_join(x = FranklinYakima_2018@data, 
                                            y = all_preds, 
                                            by = "ID")

preds_SF = rbind(WallaWalla_2015, AdamsBenton_2016, Grant_2017, FranklinYakima_2018)
preds_SF@data <- within(preds_SF@data, remove("IntlSrD", "Notes", "TRS", "RtCrpTy", "Shp_Lng", "Shap_Ar"))
preds_SF@data <- within(preds_SF@data, remove("SVM_EVI_regular_preds", 
                                              "KNN_EVI_regular_preds", 
                                              "DL_EVI_regular_prob_point9", 
                                              "RF_EVI_regular_preds", 
                                              "SVM_EVI_SG_preds", 
                                              "KNN_EVI_SG_preds",
                                              "DL_EVI_SG_prob_point9",
                                              "RF_EVI_SG_preds"))

preds_SF@data <- within(preds_SF@data, remove("SVM_NDVI_regular_preds", 
                                              "KNN_NDVI_regular_preds", 
                                              "DL_NDVI_regular_prob_point3", 
                                              "RF_NDVI_regular_preds"))


writeOGR(obj = preds_SF, 
         dsn = paste0(pred_dir, "/NDVI_SG_preds_SF/"), 
         layer="NDVI_SG_preds_SF", 
         driver="ESRI Shapefile")


preds_SF_read <- readOGR(paste0(pred_dir, "NDVI_SG_preds_SF/NDVI_SG_preds_SF.shp"),
                            layer = "NDVI_SG_preds_SF", 
                            GDAL1_integer64_policy = TRUE)

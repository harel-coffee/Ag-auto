#######################################################
#####
#####    Oct. 20, 2021
#####
#####    Goal: Filter the eastern and irrigated fields 
#####          from the intersected WSDA shapefiles of 2008-2020
#####          created by Supriya.
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
SF_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/2020_I/")

output_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")

if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Body
#

start_time <- Sys.time()

SF <- readOGR(paste0(SF_dir, "2020_I.shp"),
                     layer = "2020_I", 
                     GDAL1_integer64_policy = TRUE)

print (Sys.time() - start_time)


SF@data <- tibble::rowid_to_column(SF@data, "ID")

SF@data <- within(SF@data, 
                  remove("TRS", "TRS_1", "TRS_12", 
                         "TRS_12__14", "TRS_12__15", "TRS_12__16", 
                         "TRS_12__17", "TRS_12__18", "TRS_12__19", 
                         "TRS_12_13",  "TRS_2", "TRS_3", "TRS_4"))

SF@data <- within(SF@data, 
                 remove("Notes", "Notes_1", "Notes_1_13", "Notes_1_14",
                        "Notes_1_15",  "Notes_1_16",  "Notes_1_17" , 
                        "Notes_1_18",  "Notes_12", "Notes_2" ))

SF@data <- within(SF@data, 
                  remove("Organic", "Organic_1" , "Organic_12" ,"Organic_13",
                         "Organic_14" ,"Rot1Crop"   ,"Rot1Crop_1", "Rot1Crop_2",
                         "Rot1Crop_3" ,"Rot1Crop_4" ,"Rot1Crop_5", "Rot1Crop_6",
                         "Rot1Crop_7" ,"Rot1Crop_8" ,"Rot1Crop_9", "Rot1CropGr", 
                         "Rot1CropTy", "Rot2Crop_1" ,"Rot2Crop_2" ,"Rot2Crop_3", 
                         "Rot2Crop_4","Rot2Crop_5" ,"Rot2Crop_6", "Rot2CropGr", 
                         "Rot2CropTy" ,"Rotation_1", "RotationCr" ,"RotCropGro",
                         "RotCropT_1", "RotCropT_2", "RotCropTyp" ))

SF@data <- within(SF@data, 
                  remove("Initial_10", "Initial_11" ,"InitialS_1",
                         "InitialS_2", "InitialS_3", "InitialS_4" ,"InitialS_5",
                         "InitialS_6" ,"InitialS_7" ,"InitialS_8" ,"InitialS_9", 
                         "InitialS10" , "InitialSur"))

SF@data <- within(SF@data, 
                 remove("OBJECTI_10", "OBJECTI_11", "OBJECTID",
                        "OBJECTID_1" ,"OBJECTID_2" ,"OBJECTID_3", "OBJECTID_4", 
                        "OBJECTID_5","OBJECTID_6", "OBJECTID_7", "OBJECTID_8", 
                        "OBJECTID_9" ,"OBJECTID10" ,"SHAPE_A_10", "SHAPE_A_11" ))

SF@data <- within(SF@data, 
                 remove("County__13" , "County__14",  "County__15" , 
                        "County__16",  "County__17", "County__18",  
                        "County__19" , "County_1" ,  "County_12", 
                        "County_2" ,   "County_3"  ,  "County_4"))

SF@data <- within(SF@data, 
                  remove("TotalAcr_1", "TotalAcr_2", "TotalAcr_3", "TotalAcr_4", 
                         "TotalAcr_5", "TotalAcr_6", "TotalAcres"))

SF@data <- within(SF@data, 
                  remove("Shape_Ar_1", "Shape_Ar_2", "Shape_Ar_3", "Shape_Ar_4", 
                         "Shape_Ar_5", "Shape_Ar_6", "Shape_Ar_7", "Shape_Ar_8", 
                         "SHAPE_Ar_9", "SHAPE_Ar10", "SHAPE_Area", "SHAPE_L_10", 
                         "SHAPE_L_11", "Shape_Le_1", "Shape_Le_2", "Shape_Le_3", 
                         "Shape_Le_4", "Shape_Le_5", "Shape_Le_6", "Shape_Le_7", 
                         "Shape_Le_8", "SHAPE_Le_9", "SHAPE_Le10", "SHAPE_Leng"))

SF@data <- within(SF@data, 
                  remove("Acres_1", "Acres_1_13", "Acres_12", "Acres_2", "Acres_3"))

SF@data <- within(SF@data, 
                  remove("CoverCrop",  "CropGro_10", "CropGrou_1", "CropGrou_2", 
                         "CropGrou_3", "CropGrou_4", "CropGrou_5", "CropGrou_6", "CropGrou_7",
                         "CropGrou_8", "CropGrou_9", "CropGrou10", "CropGroup",  "CropGroup_", 
                         "CropTyp_10", "CropType","CropType_1", "CropType_2", "CropType_3", 
                         "CropType_4", "CropType_5", "CropType_6", "CropType_7",
                         "CropType_8", "CropType_9"))

SF@data <- within(SF@data, 
                 remove("DataSou_10", "DataSour_1", "DataSour_2", "DataSour_3", 
                        "DataSour_4", "DataSour_5", "DataSour_6", "DataSour_7", 
                        "DataSour_8" , "DataSour_9", "DataSour10"))

SF@data <- within(SF@data, 
                 remove("LastSur_10", "LastSur_11", "LastSurv_1", "LastSurv_2",
                        "LastSurv_3", "LastSurv_4", "LastSurv_5", "LastSurv_6", 
                        "LastSurv_7", "LastSurv_8", "LastSurv_9", "LastSurv10", "LastSurvey"))

SF@data <- within(SF@data, 
                 remove("NLCD_Cat", "NLCD_Cat_1", "NLCD_Cat_2", "NLCD_Cat_3", "NLCD_Cat_4"))


SF@data <- within(SF@data, 
                 remove("FID_WSDA_1", "FID_WSDA_2", "FID_WSDA_3", "FID_WSDA_4", 
                        "FID_WSDA_5", "FID_WSDA_6", "FID_WSDA_7", "FID_WSDA_8", 
                        "FID_WSDA_9", "FID_WSDACr"))


SF@data <- within(SF@data, 
                  remove("FID_2008_2", "FID_2010_I", "FID_2012_I", "FID_2013_I", 
                         "FID_2014_I", "FID_2015_I", "FID_2017_I", "FID_2018_I"))

SF@data$Source[SF@data$Source=="Conservation District"] <- "CD"
SF@data$DataSource[SF@data$DataSource=="Conservation District"] <- "CD"

SF@data$DataSource <- paste0(SF@data$DataSource, "-", SF@data$Source )
SF@data <- within(SF@data, remove("Source"))

pick_eastern_counties <- function(sff){
  Okanogan <- sff[grepl('Okanogan', sff$county), ]
  Chelan <- sff[grepl('Chelan', sff$county), ]

  Kittitas <- sff[grepl('Kittitas', sff$county), ]
  Yakima <- sff[grepl('Yakima', sff$county), ]

  Klickitat <- sff[grepl('Klickitat', sff$county), ]
  Douglas <- sff[grepl('Douglas', sff$county), ]

  Ferry <- sff[grepl('Ferry', sff$county), ]
  Lincoln <- sff[grepl('Lincoln', sff$county), ]

  Grant <- sff[grepl('Grant', sff$county), ]
  Benton <- sff[grepl('Benton', sff$county), ]

  Adams <- sff[grepl('Adams', sff$county), ]
  Franklin <- sff[grepl('Franklin', sff$county), ]

  Walla_Walla <- sff[grepl('Walla Walla', sff$county), ]
  Pend_Oreille <- sff[grepl('Pend Oreille', sff$county), ]

  Stevens <- sff[grepl('Stevens', sff$county), ]
  Spokane <- sff[grepl('Spokane', sff$county), ]

  Whitman <- sff[grepl('Whitman', sff$county), ]
  Garfield <- sff[grepl('Garfield', sff$county), ]

  Columbia <- sff[grepl('Columbia', sff$county), ]
  Asotin <- sff[grepl('Asotin', sff$county), ]

  sff <- raster::bind(Okanogan, Chelan, Kittitas, Yakima, Klickitat, 
                      Douglas, Benton, Ferry, Lincoln, Adams, Franklin,
                      Walla_Walla, Pend_Oreille, Stevens, Spokane, Whitman, Garfield, 
                      Columbia, Asotin, Grant)
  return(sff)

}

colnames(SF@data) <- tolower(colnames(SF@data))
colnames(SF@data)[colnames(SF@data)=="id"] <- "ID"

setnames(SF@data, 
         old=c("irrigation", "irrigati_1", "irrigati_2", "irrigati_3", 
               "irrigati_4", "irrigati_5", "irrigati_6", 
               "irrigati_7", "irrigati_8", "irrigati_9", "irrigati10",
               "irrigat_10", "irrigat_11"), 
         new=c("irr", "irr_1", "irr_2", "irr_3", 
               "irr_4", "irr_5", "irr_6", 
               "irr_7", "irr_8", "irr_9", "irr_12",
               "irr_10", "irr_11")
         )

SF@data$irr <- tolower(SF@data$irr)

SF <- pick_eastern_counties(SF)


setnames(SF@data, 
         old=c("exact_ac_1", "exact_acre", "exactacr_1", "exactacr_2", "exactacr_3",
               "exactacr_4", "exactacr_5", "exactacr_6", "exactacr_7", "exactacr_8", 
               "exactacr_9", "exactacr10", "exactacres"), 
         
         new=c("exactAc_1", "exactAc_2", "exactAc_3", "exactAc_4", "exactAc_5",
               "exactAc_6", "exactAc_7", "exactAc_8", "exactAc_19", "exactAc_10", 
               "exactAc_11", "exactAc_12", "exactAc_13")
         )

##########################################################################################
#####
#####     Compute area within each polygon here
#####
##########################################################################################

writeOGR(obj = SF,
         dsn = paste0(output_dir, "/Intersect_Eastern_irr_Acr_Cols/"), 
         layer = "Intersect_Eastern_19Cols", 
         driver = "ESRI Shapefile")


# SF@data <- within(SF@data, 
#                  remove("Exact_Ac_1", "Exact_Acre", "ExactAcr_1", "ExactAcr_2", 
#                         "ExactAcr_3", "ExactAcr_4" , "ExactAcr_5", "ExactAcr_6", "ExactAcr_7",
#                         "ExactAcr_8", "ExactAcr_9", "ExactAcr10"))

SF@data <- within(SF@data, 
                 remove("exactAc_1", "exactAc_2", "exactAc_3", "exactAc_4", "exactAc_5",
                        "exactAc_6", "exactAc_7", "exactAc_8", "exactAc_19", "exactAc_10", 
                        "exactAc_11", "exactAc_12", "exactAc_13"))

writeOGR(obj = SF,
         dsn = paste0(output_dir, "/Intersect_Eastern_IrrCols/"), 
         layer = "Intersect_Eastern_19Cols", 
         driver = "ESRI Shapefile")

write.csv(SF@data, paste0(output_dir, "data_Inter_East_19Cols.csv"), row.names = F)
SF_19cols <- copy(SF)

SF@data <- within(SF@data, 
                  remove("irr", "irr_1", "irr_2", "irr_3", "irr_4", 
                         "irr_5", "irr_6", "irr_7", "irr_8", 
                         "irr_9", "irr_10", "irr_11", "irr_12"))

SF@data <- within(SF@data, 
                  remove(acres, datasource, area))

writeOGR(obj = SF, 
         dsn = paste0(output_dir, "/Intersect_Eastern_2Cols/"), 
         layer = "Intersect_Eastern_2Cols", 
         driver = "ESRI Shapefile")

Grant <- SF[grepl('Grant', SF$county), ]
writeOGR(obj = Grant, 
         dsn = paste0(output_dir, "/intersect_Grant_2Cols/"), 
         layer = "intersect_Grant_2Cols", 
         driver = "ESRI Shapefile")


# filter_out_non_irrigated_shapefile <- function(dt){
#   dt@data$irrigation <- tolower(dt@data$irrigation)
#   dt@data$irrigation[is.na(dt@data$irrigation)] <- "na"
#   dt <- dt[!grepl('none', dt$irrigation), ]    # toss out those with None in irrigation
#   dt <- dt[!grepl('unknown', dt$irrigation), ] # toss out Unknown
#   dt <- dt[!grepl('na', dt$irrigation), ] # toss out NAs
#   return(dt)
# }

# #
# # drop anything such as "None/Sprinkler". It has None in it.
# # This is what we had decided for previous project!
# #
# SF_irr <- filter_out_non_irrigated_shapefile(SF)

# writeOGR(obj = SF_irr, 
#          dsn = paste0(output_dir, "/Intersect_EasternIrr_4Cols/"), 
#          layer = "Intersect_EasternIrr_4Cols", 
#          driver = "ESRI Shapefile")






########################################################################
A <- copy(SF_19cols@data)
A <- within(A, remove(datasource, exactacres, county, acres, ID, area))
A$irr <- tolower(A$irr)
A$irr_1 <- tolower(A$irr_1)
A$irr_2 <- tolower(A$irr_2)
A$irrCom <- paste0(A$irr, "-", A$irr_1)

A$irrCom <- A$irr ==  A$irr_1

B <- A %>% filter(irrCom == FALSE)



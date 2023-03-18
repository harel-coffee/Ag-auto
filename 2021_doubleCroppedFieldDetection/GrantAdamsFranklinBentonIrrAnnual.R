############################################################################################s
rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/", 
                   "0002_final_shapeFiles/000_Eastern_WA/")

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/NASA/parameters/"

write_dir <- paste0("/Users/hn/Documents/01_research_data/2021_doubleCroppedFieldDetection/00_SFs/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}
############################################################################################s

crop_types = read.csv(paste0(param_dir, "crop_types.csv"), as.is=TRUE)
crop_types$CropTyp = tolower(crop_types$CropTyp)

p <- crop_types %>% filter(dblCrop_potentials == "Annual") %>% 
     data.table()

h <- crop_types[grepl('hay', crop_types$CropTyp), ]
ph <- rbind(p, h)

ph <- within(ph, remove("dblCrop_potentials", "Exclude.from.CI_MikeStyle"))
ph <- within(ph, remove("CropGrp"))
ph <- unique(ph)

years <- c(2016, 2017, 2018)

for (year in years){
  WSDA <- readOGR(paste0(data_dir, "Eastern_", year, "/Eastern_", year, ".shp"),
                  layer = paste0("Eastern_", year), 
                  GDAL1_integer64_policy = TRUE)

  print (paste0("1: ", dim(WSDA@data)))
  WSDA <- filter_out_non_irrigated_shapefile(WSDA)
  print (paste0("2: ", dim(WSDA@data)))
  WSDA  <- WSDA[WSDA@data$CropTyp %in% ph$CropTyp, ]
  print (paste0("3: ", dim(WSDA@data)))

  Grant <- WSDA[grepl('Grant', WSDA$county), ]
  Benton <- WSDA[grepl('Benton', WSDA$county), ]

  Adams <- WSDA[grepl('Adams', WSDA$county), ]
  Franklin <- WSDA[grepl('Franklin', WSDA$county), ]

  WSDA <- raster::bind(Adams, Franklin, Benton, Grant)

  print (paste0("4: ", dim(WSDA@data)))

  SF_centers <- rgeos::gCentroid(WSDA, byid=TRUE)
  crs <- CRS("+proj=lcc 
              +lat_1=45.83333333333334 
              +lat_2=47.33333333333334 
              +lat_0=45.33333333333334 
              +lon_0=-120.5 +datum=WGS84")

  centroid_coord <- spTransform(SF_centers, CRS("+proj=longlat +datum=WGS84"))
  centroid_coord_dt <- data.table(centroid_coord@coords)
  setnames(centroid_coord_dt, old=c("x", "y"), new=c("cntrd_ln", "cntrd_lt"))

  centroid_coord_dt$ID <- WSDA@data$ID

  WSDA@data <- dplyr::left_join(x = WSDA@data,
                                y = centroid_coord_dt, 
                                by = "ID")
  writeOGR(obj = WSDA, 
           dsn = paste0(write_dir, "/", "AdamsFrankBentonGrant_", year, "_IrrAnn"), 
           layer = paste0("AdamsFrankBentonGrant_", year), 
           driver = "ESRI Shapefile")
}


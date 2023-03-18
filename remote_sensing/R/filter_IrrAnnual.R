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

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/", 
                    "0002_final_shapeFiles/000_Eastern_WA_IrrAnnuals/")

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

write.table(ph, 
            file = paste0(param_dir, "irrigatedAnnuals.csv"), 
            row.names=FALSE, na="", col.names=TRUE, sep=",")

years <- c(2015, 2016, 2017, 2018)
year = 2017
for (year in years){
  WSDA <- readOGR(paste0(data_dir, "Eastern_", year, "/Eastern_", year, ".shp"),
                  layer = paste0("Eastern_", year), 
                  GDAL1_integer64_policy = TRUE)

  # WSDA <- WSDA[grepl(as.character(year), WSDA$LstSrvD), ]
  WSDA <- filter_out_non_irrigated_shapefile(WSDA)
  WSDA  <- WSDA[WSDA@data$CropTyp %in% ph$CropTyp, ]


  writeOGR(obj = WSDA, 
           dsn = paste0(write_dir, "/", "Eastern_", year, "_IrrAnnuals"), 
           layer = paste0("Eastern_", year, "_correctYear"), 
           driver = "ESRI Shapefile")
}





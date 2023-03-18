
dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/05_Eastern_WA/"
years = c("2016", "2017", '2018')

yr <- "2017"
f_name <- paste0("Eastern_WA_", yr, "_70_cloud.csv")
d <- read.csv(paste0(dir, f_name), as.is=T)
yr
d$ID[1]

for (yr in years){
  cols <-  c("ID", "Acres", "BSI", "county", "CropGrp", "CropTyp", "DataSrc", "doy", "EVI", "ExctAcr", 
             "IntlSrD", "Irrigtn", "LstSrvD", "LSWI", "NDVI", "NDWI", 
             "Notes", "PSRI", "RtCrpTy", "Shap_Ar", "Shp_Lng", "system.time_start", "TRS")

  f_name <- paste0("Eastern_WA_", yr, "_70_cloud.csv")
  d <- read.csv(paste0(dir, f_name), as.is=T)
  
  d <- subset(d, select = cols)

  d$county <- as.character(d$county)
  d$TRS <- as.character(d$TRS)
  d$Irrigtn <- as.character(d$Irrigtn)
  d$IntlSrD <- as.character(d$IntlSrD)
  d$LstSrvD <- as.character(d$LstSrvD)
  d$system.time_start <- as.character(d$system.time_start)
  d$ID <- as.character(d$ID)
  d$CropGrp <- as.character(d$CropGrp)
  d$CropTyp <- as.character(d$CropTyp)
  d$DataSrc <- as.character(d$DataSrc)
  d$RtCrpTy <- as.character(d$RtCrpTy)
  
  write_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/05_Eastern_WA/non_selectors_cleaned/"
  write.csv(d, paste0(write_dir, f_name), row.names=F)

}


selector_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/05_Eastern_WA/selectors/"
f_name <- paste0("Eastern_WA_", yr, "_70_cloud.csv")
A <- read.csv(paste0(selector_dir, f_name), as.is=T)



write.csv(A, "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/05_Eastern_WA/WA_monthly_fc.csv", row.names=F)


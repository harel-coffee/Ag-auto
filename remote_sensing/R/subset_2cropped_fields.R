rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)
##################################################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/", 
                   "2Years/06_list_of_2cropped_IDs_4_website/")

SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

##################################################################################################

SF_years = c("2016", "2017", "2018")

NASS_out_options = c(FALSE)
last_survey_date_options = c(TRUE, FALSE)
double_potential_options = c(TRUE) # This must be the only option
irrigated_options = c(TRUE)


for (SF_year in SF_years){
  curr_shapefile <- readOGR(paste0(SF_dir, "Eastern_", SF_year, "/", "Eastern_", SF_year, ".shp"),
                            layer = paste0("Eastern_", SF_year), 
                            GDAL1_integer64_policy = TRUE)

  for (NASS_out in NASS_out_options){
    for (last_survey_date in last_survey_date_options){
      for (double_potential in double_potential_options){
        for (irrigated in irrigated_options){
          name_part_1 = "doubleCroppedFields"
          name_part_2 = SF_year

          # This is the only option
          if (double_potential == TRUE) {name_part_3 = "onlyAnnuals"} 
          if (irrigated        == TRUE) {name_part_4 = "JustIrr"}        else { name_part_4 = "IrrAndNonIrr"}
          if (last_survey_date == TRUE) {name_part_5 = "LastSrvCorrect"} else { name_part_5 = "LastSrvFalse"}
          if (NASS_out == TRUE)         {name_part_6 = "NASSOut"}        else { name_part_6 = "NASSIn"}
          name_part_7 = "SG_win7_Order3"

          file_name <- paste (name_part_1, name_part_2, name_part_3, 
                              name_part_4, name_part_5, name_part_6,
                              name_part_7, 
                              sep = "_")
          
          curr_data <- read.csv(paste0(data_dir, file_name, ".csv"), as.is=TRUE)

          print ("__________________")
          print (paste0("file name is: ", file_name))
          print ("__________________")
          print (paste0("curr_data$SF_year is: ", unique(curr_data$SF_year)))
          print ("__________________")
          print (head(curr_data, 2))

          filtered_SF <- curr_shapefile[curr_shapefile@data$ID %in% curr_data$ID, ]

          write_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/03_2cropped_shapeFiles_website/"
          if (dir.exists(file.path(write_dir)) == F){dir.create(path=file.path(write_dir), recursive=T)}

          output_name <- paste(name_part_2, name_part_3, name_part_4, 
                               name_part_5, name_part_6, name_part_7, 
                               sep = "_")

          writeOGR(obj = filtered_SF, 
                   dsn = paste0(write_dir, "/", output_name, "/"), 
                   layer = output_name, 
                   driver = "ESRI Shapefile")
        }
      }
    }
  }
}

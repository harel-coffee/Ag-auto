rm(list=ls())
library(dplyr)
library(data.table)


filter_out_non_irrigated_datatable <- function(dt){
  dt <- data.table(dt)
  dt$Irrigtn <- tolower(dt$Irrigtn)
  dt$Irrigtn[is.na(dt$Irrigtn)] <- "na"
  dt <- dt %>% filter(Irrigtn != "unknown") %>% data.table()
  dt <- dt %>% filter(Irrigtn != "na") %>% data.table()
  dt <- dt[!grepl('none', dt$Irrigtn), ]
  return(dt)
}

pick_eastern_counties <- function(dtt){
  eastern_counties = c("Okanogan", "Chelan", "Kittitas", "Yakima", "Klickitat", "Douglas",
                      "Grant", "Benton", "Ferry", "Lincoln", "Adams", "Franklin", "Walla Walla",
                      "Pend Oreille", "Stevens", "Spokane", "Whitman", "Garfield", "Columbia", "Asotin")
  dtt <- dtt %>% filter(county %in% eastern_counties) %>% data.table()
  return(dtt)
}

###### read the data
######
data_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
potentials <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), as.is=TRUE)

needed_cols <- c("ID", "DataSrc", "Irrigtn", "CropTyp", "LstSrvD", "county")

year = 2018
A <- data.table(read.csv(paste0(data_dir, "WSDA_DataTable_", year, ".csv" ), as.is=TRUE))
A$CropTyp <- tolower(A$CropTyp)
write.csv(A, file = paste0(data_dir, "WSDA_DataTable_", year, ".csv" ), row.names=FALSE)

# A <- pick_eastern_counties(A)

length(unique(A$ID))
nrow(A)

A <- subset(A, select=needed_cols)


important_counties = c("Grant", "Franklin", "Yakima", "Walla Walla", "Adams", "Benton", "Whitman")
A <- A %>% filter(county %in% important_counties) %>% data.table()

A <- unique(A)
nrow(A) # = length(unique(A$ID))

A <- filter_out_non_irrigated_datatable(A)
nrow(A)

A <- A %>% filter(DataSrc != "nass") %>% data.table()
dim(A) 

A <- A[grepl(as.character(year), A$LstSrvD), ]
dim(A)

A <- A %>% filter(CropTyp %in% potentials$Crop_Type) %>% data.table()
dim(A) 

write.csv(A, 
          file = paste0(param_dir, "Fields_", year, "_NoNass_OnlyIrri_LastSurvey_Potentials.csv"), 
          row.names=FALSE)



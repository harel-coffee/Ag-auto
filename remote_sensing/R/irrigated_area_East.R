rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)


data_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"
year = 2017

f_name = paste0("WSDA_DataTable_", year, ".csv")
WSDA_2017 <- read.csv(paste0(data_dir, f_name))

East_2017 <- pick_eastern_counties(WSDA_2017)
East_2017_irrigated <- filter_out_non_irrigated_datatable(East_2017)
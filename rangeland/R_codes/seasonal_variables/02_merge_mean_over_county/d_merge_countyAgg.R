
library(data.table)
library(dplyr)
options(dplyr.summarise.inform = FALSE)

binary_source_path = "/home/h.noorazar/Sid/sidFabio/read_binary_core.R" # Kamiak
source(binary_source_path)
options(digit=9)
options(digits=9)

source_path = "/home/h.noorazar/rangeland/rangeland_core.R"
source(source_path)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 9)

start_time <- Sys.time()
#
#
#   Years:      1992-2017 (Gridmet data are from 1979)
#   Grid count: 146229 in 25 states of interest.
#   Seasons:    Season 1: Jan - Mar
#               Season 2: Apr - Jul
#               Season 3: Aug - Sep
#               Season 4: Oct - Dec
#
#
########################################################################
######
######      Directories
######
########################################################################
data_base <- "/data/project/agaid/h.noorazar/rangeland/"
param_dir <- paste0(data_base, "parameters/")

in_dir <- paste0(data_base, "/seasonal_variables/01_mean_over_county/")
out_dir<- paste0(data_base, "/seasonal_variables/02_merged_mean_over_county/")
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)
}
########################################################################
######
######      Body
######
########################################################################

all_data <- data.table()
all_files <- list.files(path=in_dir, pattern=".csv", full.names=FALSE)

for (a_file in all_files){
  df = data.table(read.csv(paste0(in_dir, a_file)))
  all_data = rbind(all_data, df)
}

write.csv(all_data, 
          file = paste0(out_dir, "countyMean_seasonalVars.csv"), 
          row.names=FALSE)

grids_25states <- data.table(read.csv(paste0(param_dir, "Bhupi_25states_clean.csv")))
grids_25states <- grids_25states[, c("state", "county", "county_fips")]
grids_25states = unique(grids_25states)
all_data <- dplyr::left_join(x = all_data, y = grids_25states, by = c("state", "county"))
write.csv(all_data, 
          file = paste0(out_dir, "countyMean_seasonalVars_wFips.csv"), 
          row.names=FALSE)


# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)





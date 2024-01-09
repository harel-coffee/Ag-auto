
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
######      Terminal/shell/bash arguments                      
######
########################################################################

args = commandArgs(trailingOnly=TRUE)
a_state = args[1]
a_state = gsub("_", " ", a_state)
# model_type = args[2] # observed, modeled_hist, or GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
# model_type = gsub("-", "", model_type)

########################################################################
######
######      Directories
######
########################################################################
data_base <- "/data/project/agaid/h.noorazar/rangeland/"
param_dir <- paste0(data_base, "parameters/")

in_dir <- paste0(data_base, "seasonal_variables/00_gridwise/")
out_dir = paste0(data_base, "/seasonal_variables/01_mean_over_county/")
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)
}
########################################################################
######
######      Body
######
########################################################################
#
#   Define parameters
#
grids_25states <- data.table(read.csv(paste0(param_dir, "Bhupi_25states_clean.csv")))
curr_state_grids = grids_25states[grids_25states$state==a_state]

####
#### to detect how many rows we need
#### for pre-allocation
####

df = read.csv(paste0(in_dir, curr_state_grids$grid[1], ".csv"))
# grid_vec = strsplit(curr_state_grids$grid[1], "_")[[1]]
# grid_ = paste(grid_vec[2], grid_vec[3], sep = "_")
df$grid = curr_state_grids$grid[1]

# each location has this # of rows. We need to multiply that by number of location per state.
nYears_times_nSeasons = nrow(df) 
total_nRows = nYears_times_nSeasons * length(unique(curr_state_grids$grid))

all_data <- setNames(data.table(matrix(nrow = total_nRows, ncol = ncol(df))), colnames(df))
# Change class of certain columns
# columns_classType = as.vector(sapply(df , class))
for (a_col in colnames(df)){
  if (typeof(df[a_col][[1]][1])=="integer"){
    all_data[ , (a_col) := lapply(.SD, as.integer), .SDcols = a_col]
    } else if (typeof(df[a_col][[1]][1])=="double"){
      all_data[ , (a_col) := lapply(.SD, as.double), .SDcols = a_col]
    } else if (typeof(df[a_col][[1]][1])=="character"){
    all_data[ , (a_col) := lapply(.SD, as.character), .SDcols = a_col]
  }
}

top_pointer = 1
for (a_file in curr_state_grids$grid){
  curr_loc_dt <- read.csv(paste0(in_dir, a_file, ".csv"))
  curr_loc_dt$grid = a_file

  low_pointer = top_pointer + nrow(curr_loc_dt) - 1
  all_data[top_pointer: low_pointer] = curr_loc_dt
  top_pointer = low_pointer + 1
}

all_data = na.omit(all_data)
all_data <- dplyr::left_join(x = all_data, y = curr_state_grids, by = c("grid"))


# all_data %>%
#      group_by(year, season) %>%
#      summarise_at(.vars=c("avg_Tavg", "total_precip"), mean) %>% 
#      data.table()


agg_data <- all_data[, list(countyMean_avg_Tavg=mean(avg_Tavg), 
                            countyMean_total_precip=mean(total_precip)), 
                       by=c("state", "county", "year", "season")]


write.csv(agg_data, 
          file = paste0(out_dir, gsub(" ", "_", a_state), "_countyMean_seasonalVars.csv"), 
          row.names=FALSE)

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)





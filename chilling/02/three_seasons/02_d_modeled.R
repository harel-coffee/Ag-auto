.libPaths("/data/hydro/R_libs35")
.libPaths()
library(plyr)
library(lubridate)
library(purrr)
library(tidyverse)

source_1 <- "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_1)
options(digit=9)
options(digits=9)

print("does this look right?")
getwd()
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                     Terminal arguments                           ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
model_type = args[1]
overlap_type = args[2]
season_start = args[3]

######################################################################
##                                                                  ##
##           non_overlap    vs    overlap                           ##
##           dynamic        vs    utah                              ##
##                                                                  ##
######################################################################
# 2. Script setup ----------------------------------------------------

chill_out = "/data/hydro/users/Hossein/chill/data_by_core/"
main_out <- file.path(chill_out, model_type, "02_with_May", season_start)

if (overlap_type == "non_overlap" ){
  main_out <- file.path(main_out, "non_overlap/")
  } else if (overlap_type == "overlap" ) {
    main_out <- file.path(main_out, "overlap/")
}

# Create a figures-specific output pathway if it doesn't exist
if (dir.exists(file.path(main_out)) == F){
  dir.create(path = main_out, recursive = T)}

# 3. Some set up ----------------------------------------------------

# List of filenames
the_dir <- dir()
print ("line 53")
print (the_dir)

# Remove file names that aren't data, if they exist
the_dir <- the_dir[grep(pattern = "chill_output_data", x = the_dir)]

# Pre-allocate lists to be used
param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
param_dir <- file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, 
                                        "file_list.txt"), 
                          header = F)
local_files <- as.vector(local_files$V1)
no_sites <- length(local_files)

data_list_hist <- vector(mode = "list", length = no_sites)
data_list_F <- vector(mode = "list", length = no_sites)

# Check whether historical data or not
hist <- basename(getwd()) == "historical"

# 5. Iterate through files and process ---------------------------
# If historical data, then run a simpler data cleaning routine

if(hist){
  # 5a. Iterate through historical files -------------------------
  for(i in 1:length(the_dir)){
    file <- read.table(file = the_dir[i], header = T,
                       colClasses = c("factor", "numeric", 
                                      "numeric", "numeric",
                                      "numeric", "numeric"))
    file <- na.omit(file)
    names(data_list_hist)[i] <- the_dir[i]
    file <- na.omit(file)
    # Append it to a list following some processing
    data_list_hist[[i]] <-  threshold_func(file, data_type="modeled")
    rm(file)
  }

  # 5b. Process gathered historical data ------------------------------
  # Get medians for each location during historical period
  summary_data_historical <- get_medians(data_list_hist)
  
  # Briefly want to export the raw data from 
  # the lists for use in other figs
  data_historical <- ldply(data_list_hist, function(x) data.frame(x))

  data_historical$year <- as.numeric(substr(x=data_historical$chill_season,
                                            start = 12, stop = 15))
  data_historical$model <- basename(dirname(getwd()))
  data_historical$emission <- basename(getwd())
  data_historical$lat <- as.numeric(substr(x=data_historical$.id, 
                                           start=19, stop=26))
  data_historical$long<- as.numeric(substr(x=data_historical$.id, 
                                           start=28, stop=37))
  data_historical <- unique(data_historical)

  rm(data_list_hist)
  # .id row contains originating filename of this data
  data_historical <- add_time_periods_model(data_historical)
  print ("this is historical modeled, needs to have time period")
  print(colnames(data_historical))
  data_historical <- na.omit(data_historical)
  write.table(x = data_historical,
              file = file.path(main_out,
                               paste0("summary_",
                     # model name
                     gsub("-", "_", basename(dirname(getwd()))),
                     "_",
                     basename(getwd()), # emission
                     ".txt")), row.names = F)
    
  rm(data_historical)
    
  # Grab lat/long
  summary_data_historical <- grab_coord(summary_data_historical)
  summary_data_historical$model <- basename(dirname(getwd()))
  summary_data_historical$emission <- basename(getwd())
  summary_data_historical <- add_time_periods_model(summary_data_historical)
  summary_data_historical <- na.omit(summary_data_historical)
  out <- paste0(main_out, "/stats/")
  if (dir.exists(file.path(out)) == F){
     dir.create(path = out, recursive = T)}

  write.table(x=summary_data_historical,
              file = file.path(out, paste0("/summary_stats_",
                     # model name
                     gsub("-", "_",basename(dirname(getwd()))),
                     "_", basename(getwd()), # emission
                     ".txt")), row.names = F)
  
  # 5c. Iterate through future files ------------------------
  } else {
     for(i in 1:length(the_dir)){
         file <- read.table(file = the_dir[i], header = T,
                          colClasses = c("factor", "numeric", 
                                         "numeric", "numeric",
                                         "numeric", "numeric"))
         names(data_list_F)[i] <- the_dir[i]
         file <- na.omit(file)
         data_list_F[[i]] <- process_data_non_overlap(file, 
                               time_period="doesNotMatter")
         rm(file)
     }
  
  # 5d. Process gathered future data -------------------------
  print("line 145")
  # Apply this function to a list and spit out a dataframe
  summary_data_F <- get_medians(data_list_F)
  print("line 148")
  # Briefly want to export the raw data from the lists 
  # for use in other figs
  all_years <- ldply(data_list_F, function(x) data.frame(x))
  print("line 151")
  all_years$year <- as.numeric(substr(x = all_years$chill_season,
                                      start = 7, stop = 10))
  all_years$model <- basename(dirname(getwd()))
  all_years$emission <- basename(getwd())
  print("line 156")
  all_years$lat = as.numeric(substr(x = all_years$.id, 
                                    start = 19, stop = 26))
  all_years$long = as.numeric(substr(x = all_years$.id, 
                                     start = 28, stop = 37))
  print("line 159")
  all_years <- unique(all_years)
  all_years <- add_time_periods_model(all_years)
  print("line 162")
  # .id row contains originating filename of this data
  all_years <- na.omit(all_years)
  write.table(x = all_years,
              file = file.path(main_out,
                               paste0("summary_",
                     # model name
                     gsub( "-", "_", basename(dirname(getwd()))),
                     "_", basename(getwd()), # emission
                     ".txt")), row.names = F)
  rm(all_years)
   
  # Grab lat/long
  summary_data_F <- grab_coord(summary_data_F)
  summary_data_F$model <- basename(dirname(getwd()))
  summary_data_F$emission <- basename(getwd())
  summary_data_F <- add_time_periods_model(summary_data_F)
  summary_data_F <- na.omit(summary_data_F)
  out <- paste0(main_out, "/stats/")
  if (dir.exists(file.path(out)) == F){
     dir.create(path = out, recursive = T)}
  write.table(x = summary_data_F,
              file = file.path(out, paste0("/summary_stats_",
                     # model name
                     gsub("-", "_", basename(dirname(getwd()))),
                     "_",
                     basename(getwd()), # emission
                     ".txt")),
              row.names = F) 
  print ("line 182")
}
print (main_out)
# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)




# It is not clear why the hell did stopped working
# Things is the same as before. But somehow, when
# the data is broken to 3 time periods, then SUM_J1 becomes problematic
# something similar to this: 
# https://stackoverflow.com/questions/50781829/column-rate-must-be-length-1-a-summary-value-not-22906
# So, in the driver and core, I got rid of breaking down stuff into three time periods: 
# the function is core is process_data_non_overlap(.) and I wrote the new driver.
#
#
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(plyr)
library(lubridate)
library(purrr)
library(tidyverse)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)
options(digit=9)
options(digits=9)
# Check current folder
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
# 2. Script setup ---------------------------------------------------------

chill_out = "/data/hydro/users/Hossein/chill/data_by_core/"

main_out <- file.path(chill_out, model_type, "02", season_start)
if (overlap_type == "non_overlap" ){
  main_out <- file.path(main_out, "non_overlap/")
  } else if (overlap_type == "overlap" ) {
    main_out <- file.path(main_out, "overlap/")
}

# Create a figures-specific output pathway if it doesn't exist
if (dir.exists(file.path(main_out)) == F) {
  dir.create(path = main_out, recursive = T)
}

# 3. Some set up --------------------------------------------------

# List of filenames
the_dir <- dir()

# Remove file names that aren't data, if they exist
the_dir <- the_dir[grep(pattern = "chill_output_data", x = the_dir)]

# Pre-allocate lists to be used
param_dir = file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, "file_list.txt"), 
                          header = F)
local_files <- as.vector(local_files$V1)
no_sites <- length(local_files)

data_list_hist <- vector(mode = "list", length = no_sites)
data_list_F0 <- vector(mode = "list", length = no_sites) # 2005-2024
data_list_F1 <- vector(mode = "list", length = no_sites) # 2025-2050
data_list_F2 <- vector(mode = "list", length = no_sites) # 2051-2075
data_list_F3 <- vector(mode = "list", length = no_sites) # 2076-2099

# Check whether historical data or not
hist <- basename(getwd()) == "historical"

# 5. Iterate through files and process ------------------------------------
# If historical data, then run a simpler data cleaning routine

if(hist){
    # 5a. Iterate through historical files ----------------------------------
    for(i in 1:length(the_dir)){
      file <- read.table(file = the_dir[i],
                         header = T,
                         colClasses = c("factor", "numeric", "numeric", "numeric",
                                        "numeric", "numeric"))
      
      names(data_list_hist)[i] <- the_dir[i]
      
      # Append it to a list following some processing
      data_list_hist[[i]] <-  threshold_func(file, data_type="modeled")
      rm(file)
    }

    # 5b. Process gathered historical data ------------------------------------
    # Get medians for each location during historical period
    summary_data_historical <- get_medians(data_list_hist)
    
    # Briefly want to export the raw data from the lists for use in other figs
    data_historical <- ldply(data_list_hist, function(x) data.frame(x))

    data_historical$year <- as.numeric(substr(x = data_historical$chill_season,
                                              start = 7, stop = 10))
    data_historical$model <- basename(dirname(getwd()))
    data_historical$scenario <- basename(getwd())
    data_historical$lat <- as.numeric(substr(x = data_historical$.id, start=19, stop=26))
    data_historical$long<- as.numeric(substr(x = data_historical$.id, start=28, stop=37))
    data_historical <- unique(data_historical)
    
    # No longer needed
    rm(data_list_hist)
    print("line 112 of d modeled")
    # .id row contains originating filename of this data
    write.table(x = data_historical,
                file = file.path(main_out,
                                 paste0("summary_",
                                        # model name
                                        gsub("-", "_", basename(dirname(getwd()))),
                                        "_",
                                        basename(getwd()), # scenario
                                        ".txt")),
                row.names = F)
    
    rm(data_historical)
    
    # Grab lat/long
    summary_data_historical <- grab_coord(summary_data_historical)
    
    summary_data_historical$model <- basename(dirname(getwd()))
    summary_data_historical$scenario <- basename(getwd())

    write.table(x = summary_data_historical,
                file = file.path(main_out,
                                 paste0("summary_stats_",
                                        # model name
                                        gsub("-", "_",basename(dirname(getwd()))),
                                        "_",
                                        basename(getwd()), # scenario
                                        ".txt")),
                row.names = F)

    # If future data, then proceed with decadal calculations:
    
    # 5c. Iterate through future files ----------------------------------------
  } else {
  print("line 144")
  print(length(the_dir))
  for(i in 1:length(the_dir)){
    file <- read.table(file = the_dir[i],
                       header = T,
                       colClasses = c("factor", "numeric", "numeric", "numeric",
                                      "numeric", "numeric"))
    print("line 151 of d modeled")
    print(dim(file))
    if (overlap_type == "overlap"){
      # 2005-2025
      data_list_F0[[i]] <- process_data(file, time_period="2015")
      names(data_list_F0)[i] <- the_dir[i]

      # 2040s
      data_list_F1[[i]] <- process_data(file, time_period="2040")
      names(data_list_F1)[i] <- the_dir[i]
      
      # 2060s
      data_list_F2[[i]] <- process_data(file, time_period="2060")
      names(data_list_F2)[i] <- the_dir[i]
      
      # 2080s
      data_list_F3[[i]] <- process_data(file, time_period="2080")
      names(data_list_F3)[i] <- the_dir[i]

      rm(file) 
    } else if (overlap_type == "non_overlap"){
      print("line 170")
      # 2005_2024
      data_list_F0[[i]] <- process_data_non_overlap(file, time_period="2005-2024")
      names(data_list_F0)[i] <- the_dir[i]
      print(paste0("line 176", dim(data_list_F0[[i]]))

      # 2025_2050
      data_list_F1[[i]] <- process_data_non_overlap(file, time_period="2025-2050")
      print(paste0("line 176", dim(data_list_F1[[i]]))
      names(data_list_F1)[i] <- the_dir[i]
 
      # 2051_2075
      data_list_F2[[i]] <- process_data_non_overlap(file, time_period="2051-2075")
      names(data_list_F2)[i] <- the_dir[i]
      print(paste0("line 176", dim(data_list_F2[[i]]))

      # 2076_2100
      data_list_F3[[i]] <- process_data_non_overlap(file, time_period="2076-2100")
      names(data_list_F3)[i] <- the_dir[i]
      print(paste0("line 176", dim(data_list_F3[[i]]))

      rm(file)
    }
}
  
  # 5d. Process gathered future data ----------------------------------------
  
  # Apply this function to a list and spit out a dataframe
  summary_data_F0 <- get_medians(data_list_F0)
  summary_data_F1 <- get_medians(data_list_F1)
  summary_data_F2 <- get_medians(data_list_F2)
  summary_data_F3 <- get_medians(data_list_F3)
  
  # Briefly want to export the raw data from the lists for use in other figs
  dataF0 <- ldply(data_list_F0, function(x) data.frame(x))
  dataF1 <- ldply(data_list_F1, function(x) data.frame(x))
  dataF2 <- ldply(data_list_F2, function(x) data.frame(x))
  dataF3 <- ldply(data_list_F3, function(x) data.frame(x))
  
  all_years <- bind_rows(dataF0, dataF1, dataF2, dataF3)

  # No longer needed
  rm(list = c("data_list_F0", "data_list_F1", "data_list_F2", "data_list_F3"))

  all_years$year <- as.numeric(substr(x = all_years$chill_season,
                                      start = 7, stop = 10))
  all_years$model <- basename(dirname(getwd()))
  all_years$scenario <- basename(getwd())
  all_years$lat <- as.numeric(substr(x = all_years$.id, start = 19, stop = 26))
  all_years$long <- as.numeric(substr(x = all_years$.id, start = 28, stop = 37)) 
  all_years <- unique(all_years)
 
  # .id row contains originating filename of this data
  write.table(x = all_years,
              file = file.path(main_out,
                               paste0("summary_",
                                      # model name
                                      gsub( "-", "_", basename(dirname(getwd()))),
                                      "_",
                                      basename(getwd()), # scenario
                                      ".txt")),
              row.names = F)
  
  rm(all_years)
   
  # Grab lat/long
  summary_data_F0 <- grab_coord(summary_data_F0)
  summary_data_F1 <- grab_coord(summary_data_F1)
  summary_data_F2 <- grab_coord(summary_data_F2)
  summary_data_F3 <- grab_coord(summary_data_F3)
  
  # Combine dfs for plotting ease
  if (overlap_type == "non_overlap"){
    summary_data_F0 <- summary_data_F0 %>% mutate(time_period = "2005-2024")
    summary_data_F1 <- summary_data_F1 %>% mutate(time_period = "2025-2050")
    summary_data_F2 <- summary_data_F2 %>% mutate(time_period = "2051-2075")
    summary_data_F3 <- summary_data_F3 %>% mutate(time_period = "2076-2100")
  } else if (overlap_type == "overlap"){
    summary_data_F0 <- summary_data_F0 %>% mutate(time_period = "2015s")
    summary_data_F1 <- summary_data_F1 %>% mutate(time_period = "2040s")
    summary_data_F2 <- summary_data_F2 %>% mutate(time_period = "2060s")
    summary_data_F3 <- summary_data_F3 %>% mutate(time_period = "2080s")
  }
  
  summary_data_comb <- bind_rows(summary_data_F0,
                                 summary_data_F1,
                                 summary_data_F2,
                                 summary_data_F3)
  
  summary_data_comb$model <- basename(dirname(getwd()))
  summary_data_comb$scenario <- basename(getwd())
 
  write.table(x = summary_data_comb,
              file = file.path(main_out,
                               paste0("summary_stats_",
                                      # model name
                                      gsub("-", "_", basename(dirname(getwd()))),
                                      "_",
                                      basename(getwd()), # scenario
                                      ".txt")),
              row.names = F)   
}
# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)



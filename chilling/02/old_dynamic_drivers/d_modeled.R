
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(plyr)
library(lubridate)
library(purrr)
library(tidyverse)
options(digit=9)
options(digits=9)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)

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
######################################################################
##                                                                  ##
##           non_overlap    vs    overlap                           ##
##           dynamic        vs    utah                              ##
##                                                                  ##
######################################################################
# 2. Script setup ---------------------------------------------------------

chill_out = "/data/hydro/users/Hossein/chill/data_by_core/"

main_out <- file.path(chill_out, model_type, "02/")

if (overlap_type == "non_overlap" ){
  main_out <- file.path(main_out, "/non_overlap/")
  } else if (overlap_type == "overlap" ) {
    main_out <- file.path(main_out, "/overlap/")
}

print (model_type)
print (overlap_type)
print (main_out)
# Create a figures-specific output pathway if it doesn't exist
if (dir.exists(file.path(main_out)) == F) {
  dir.create(path = main_out, recursive = T)
}

# 3. Some set up ----------------------------------------------------------

# List of filenames
the_dir <- dir()

# Remove file names that aren't data, if they exist
the_dir <- the_dir[grep(pattern = "chill_output_data", x = the_dir)]


# Pre-allocate lists to be used
no_sites <- 645
data_list_hist <- vector(mode = "list", length = no_sites)
data_list_F1 <- vector(mode = "list", length = no_sites)
data_list_F2 <- vector(mode = "list", length = no_sites)
data_list_F3 <- vector(mode = "list", length = no_sites)

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
                                            start = 12, stop = 15))
  data_historical$model <- basename(dirname(getwd()))
  data_historical$scenario <- basename(getwd())
  data_historical$lat = as.numeric(substr(x = data_historical$.id,
                                          start = 19, stop = 26))
  data_historical$long = as.numeric(substr(x = data_historical$.id,
                                           start = 28, stop = 37))
  data_historical <- unique(data_historical)
  
  # No longer needed
  rm(data_list_hist)
  
  # .id row contains originating filename of this data
  write.table(x = data_historical,
              file = file.path(main_out,
                               paste0("summary_",
                                      gsub("-", "_", basename(dirname(getwd()))), # model name
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
                                      gsub("-", "_",basename(dirname(getwd()))), # model name
                                      "_",
                                      basename(getwd()), # scenario
                                      ".txt")),
              row.names = F)

  # If future data, then proceed with decadal calculations:
  
  # 5c. Iterate through future files ----------------------------------------
} else {
  for(i in 1:length(the_dir)){
    file <- read.table(file = the_dir[i],
                       header = T,
                       colClasses = c("factor", "numeric", "numeric", "numeric",
                                      "numeric", "numeric"))

    if (overlap_type == "overlap"){
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
      # 2025_2050s
      data_list_F1[[i]] <- process_data_non_overlap(file, time_period="2025_2050")
      names(data_list_F1)[i] <- the_dir[i]
 
      # 2051_2075s
      data_list_F2[[i]] <- process_data_non_overlap(file, time_period="2051_2075")
      names(data_list_F2)[i] <- the_dir[i]

      # 2076_2100s
      data_list_F3[[i]] <- process_data_non_overlap(file, time_period="2076_2100")
      names(data_list_F3)[i] <- the_dir[i]

      rm(file)
    }
  }
  
  # 5d. Process gathered future data ----------------------------------------
  
  # Apply this function to a list and spit out a dataframe
  summary_data_F1 <- get_medians(data_list_F1) 
  summary_data_F2 <- get_medians(data_list_F2)
  summary_data_F3 <- get_medians(data_list_F3)
  
  # Briefly want to export the raw data from the lists for use in other figs
  dataF1 <- ldply(data_list_F1, function(x) data.frame(x))
  dataF2 <- ldply(data_list_F2, function(x) data.frame(x))
  dataF3 <- ldply(data_list_F3, function(x) data.frame(x))
  
  all_years <- bind_rows(dataF1, dataF2, dataF3)

  all_years$year <- as.numeric(substr(x = all_years$chill_season,
                                      start = 12, stop = 15))
  all_years$model <- basename(dirname(getwd()))
  all_years$scenario <- basename(getwd())
  all_years$lat = as.numeric(substr(x = all_years$.id, start = 19, stop = 26))
  all_years$long = as.numeric(substr(x = all_years$.id, start = 28, stop = 37))
  all_years <- unique(all_years)

  # No longer needed
  rm(list = c("data_list_F1", "data_list_F2", "data_list_F3"))
   
  # .id row contains originating filename of this data
  write.table(x = all_years,
              file = file.path(main_out,
                               paste0("summary_",
                                      gsub( "-", "_", basename(dirname(getwd()))), # model name
                                      "_",
                                      basename(getwd()), # scenario
                                      ".txt")),
              row.names = F)
  
  rm(all_years)
   
  # Grab lat/long
  summary_data_F1 <- grab_coord(summary_data_F1)
  summary_data_F2 <- grab_coord(summary_data_F2)
  summary_data_F3 <- grab_coord(summary_data_F3)
  
  # Combine dfs for plotting ease
  if (overlap_type == "non_overlap"){
    summary_data_F1 <- summary_data_F1 %>% mutate(time_period = "2025_2050")
    summary_data_F2 <- summary_data_F2 %>% mutate(time_period = "2051_2075")
    summary_data_F3 <- summary_data_F3 %>% mutate(time_period = "2076_2100")
  } else if (overlap_type == "overlap"){
    summary_data_F1 <- summary_data_F1 %>% mutate(time_period = "2040's")
    summary_data_F2 <- summary_data_F2 %>% mutate(time_period = "2060's")
    summary_data_F3 <- summary_data_F3 %>% mutate(time_period = "2080's")
  }
  
  summary_data_comb <- bind_rows(summary_data_F1,
                                 summary_data_F2,
                                 summary_data_F3)
  
  summary_data_comb$model <- basename(dirname(getwd()))
  summary_data_comb$scenario <- basename(getwd())
 
  write.table(x = summary_data_comb,
              file = file.path(main_out,
                               paste0("summary_stats_",
                                      gsub("-", "_", basename(dirname(getwd()))), # model name
                                      "_",
                                      basename(getwd()), # scenario
                                      ".txt")),
              row.names = F)   
}
# How long did it take?
end_time <- Sys.time()

print( end_time - start_time)



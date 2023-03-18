.libPaths("/data/hydro/R_libs35")
.libPaths()
library(chillR)
library(tidyverse)
library(lubridate)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)

options(digit=9)
options(digits=9)


# 1. Prep binary conversion function --------------------------------------
read_binary_8 <- function(file_name, file_path, hist){
  if (hist) {
    start_year <- 1979
    end_year <- 2015
  }

  if (!hist) {
    start_year <- 2006
    end_year <- 2099
  }

  ## create year month day values ##
  create_ymdvalues <- function(data_start_year, data_end_year) {
    Years <- seq(data_start_year, data_end_year)
    nYears <- length(Years)
    daycount_in_year <- 0
    moncount_in_year <- 0
    yearrep_in_year <- 0

    for (i in 1:nYears){
      ly <- leap_year(Years[i])

      if (ly == TRUE){
        days_in_mon <- c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
      }

      else{
        days_in_mon <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
      }

      for (j in 1:12){
        daycount_in_year <- c(daycount_in_year, seq(1, days_in_mon[j]))
        moncount_in_year <- c(moncount_in_year, rep(j, days_in_mon[j]))
        yearrep_in_year <- c(yearrep_in_year, rep(Years[i], days_in_mon[j]))
      }
    }

    daycount_in_year <- daycount_in_year[-1] #delete the leading 0
    moncount_in_year <- moncount_in_year[-1]
    yearrep_in_year <- yearrep_in_year[-1]
    ymd <- cbind(yearrep_in_year, moncount_in_year, daycount_in_year)
    colnames(ymd) <- c("year", "month", "day")
    return(ymd)
  }

  ## read binary data and add dates ##
  readbinarydata_addmdy <- function(filename, ymd) {
    Nofvariables <- 8 # number of variables or column in the forcing data file
    Nrecords <- nrow(ymd)
    ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
    fileCon  <-  file(filename, "rb")
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables,
                    endian = "little")
    dataM <- matrix(0, Nrecords, 8)
    k <- 1
    dataM[1:Nrecords, 1] <- temp[ind] / 40.00  #precip data
    dataM[1:Nrecords, 2] <- temp[ind + 1] / 100.00  #Max temperature data
    dataM[1:Nrecords, 3] <- temp[ind + 2] / 100.00  #Min temperature data
    dataM[1:Nrecords, 4] <- temp[ind + 3] / 100.00  #Wind speed data
    dataM[1:Nrecords, 5] <- temp[ind + 4] / 10000.00  #SPH
    dataM[1:Nrecords, 6] <- temp[ind + 5] / 40.00  #SRAD
    dataM[1:Nrecords, 7] <- temp[ind + 6] / 100.00  #Rmax
    dataM[1:Nrecords, 8] <- temp[ind + 7] / 100.00  #RMin
    AllData <- cbind(ymd, dataM)
    # calculate daily GDD  ...what? There doesn't appear to be any GDD work?
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin",
                           "windspeed", "SPH", "SRAD", "Rmax", "Rmin")
    close(fileCon)
    return(AllData)
  }

  # create date ranges #
  ymd_file <- create_ymdvalues(start_year, end_year)

  # read data #
  readbinarydata_addmdy(file_path, ymd_file)
}

# 2. Pre-processing prep --------------------------------------------------
param_dir = "/home/hnoorazar/chilling_codes/parameters/"

# 2a. Only use files in geographic locations we're interested in
local_files <- read.delim(file = paste0(param_dir, "limited_locations.txt"),
                          header = F)
local_files <- as.vector(local_files$V1)

# 2b. Note if working with a directory of historical data

hist <- TRUE
# 2c. If needed, make fastscratch folder matching the name of the 
#     current directory

# Define main output path
main_out <- file.path("/data/hydro/users/Hossein/chill/7_time_intervals/observed/")

if (dir.exists(main_out) == F) {
  dir.create(path = main_out, recursive = T)
}

# 2d. Prep list of files for processing

# get files in current folder
dir_con <- dir()

# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "data_",
                        x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

print(dir_con)
# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

for(file in dir_con){
  # 3a. read in binary meteorological data file from specified path
  met_data <- read_binary_8(file_path = file,
                            hist = hist)

  # I make the assumption that lat always has same number of decimal points
  lat <- as.numeric(substr(x = file, start = 6, stop = 13))

  # data frame required
  met_data <- as.data.frame(met_data)
  
  # 3b. Clean it up
  # rename needed columns
  met_data <- met_data %>%
              select(-c(precip, windspeed, SPH, SRAD, Rmax, Rmin)) %>%
              data.frame()

  data.table::setnames(met_data, old=c("year","month", "day", "tmax", "tmin"), 
                                 new=c("Year", "Month", "Day", "Tmax", "Tmin"))
  
  # 3c. Get hourly interpolation
  # generate hourly data
  met_hourly <- stack_hourly_temps(weather = met_data,
                                   latitude = lat)

  # save only the necessary list item
  met_hourly <- met_hourly[[1]]
  print ("line 159 of driver")
  print (colnames(met_hourly))
  data.table::setnames(met_hourly, new=c("year","month", "day", "tmax", "tmin"), 
                                 old=c("Year", "Month", "Day", "Tmax", "Tmin"))
  rm(met_data)
  print (colnames(met_hourly))
  print (dim(met_hourly))
  # 3d. Run the chill accumulation model and sum up by day
  # we want this on a seasonal basis specific to chill
  met_hourly <- put_chill_season(met_hourly, "sept")

  # 3e. Save output
  saveRDS(met_hourly, paste0(main_out, "/met_hourly_", file, ".rds"))

  # Remove objects not needed in future iterations
  rm(met_hourly)
}

# How long did it take?
end_time <- Sys.time()

print( end_time - start_time)


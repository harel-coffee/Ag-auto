options(digit=9)
options(digits=9)

.libPaths("/data/hydro/R_libs35") # remove in Kamiak
.libPaths() # remove in Kamiak
library(data.table)
# library(lubridate)
library(chron)
library(dplyr)

# Define function for reading binary 8-col files

###### The modeled historical is in /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/
###### modeled historical is equivalent to having 4 variables, and years 1950-2005
######
###### The observed historical is in 
######  /data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016
###### observed historical is equivalent to having 8 variables, and years 1979-2016
######

###### YOU MUST be careful with start and end years anywhere in this code.
###### The start and end year are not in the binary files (I am faily certain.)
###### So, if you start 2100 and end on 2120, it will create a calendar using those
###### and then read the data and populate the data table and you have observed the future!!!

read_binary_new <- function(file_path, no_vars, start_year, end_year){
  ymd_file <- create_ymdvalues(start_year, end_year)
  data <- read_binary_addmdy(file_path, ymd_file, no_vars)
  return(data.table(data))
}

read_binary <- function(file_path, hist, no_vars){
  if (no_vars == 8){
    if(hist == FALSE) {
      stop("No. variables is 8 but hist is FALSE!!!")
    }
    start_year <- 1979
    end_year <- 2016
   } else if (no_vars == 4) {
     if (hist){
       start_year <- 1950
       end_year <- 2005
       } else {
          start_year <- 2006
          end_year <- 2099
      }
  }
  ymd_file <- create_ymdvalues(start_year, end_year)
  data <- read_binary_addmdy(file_path, ymd_file, no_vars)
  return(data.table(data))
}

read_binary_addmdy <- function(filename, ymd, no_vars){
  if (no_vars==4){
    return(read_binary_addmdy_4var(filename, ymd))
  } else {return(read_binary_addmdy_8var(filename, ymd))}
}

read_binary_addmdy_8var <- function(filename, ymd){
  Nofvariables <- 8 # number of variables or column in the forcing data file
  Nrecords <- nrow(ymd)
  ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
  fileCon  <-  file(filename, "rb")
  temp <- readBin(fileCon, integer(), size=2, n=Nrecords*Nofvariables,
                  endian = "little")
  dataM <- matrix(0, Nrecords, 8)
  dataM[1:Nrecords, 1] <- temp[ind] / 40.00         # precip data
  dataM[1:Nrecords, 2] <- temp[ind + 1] / 100.00    # Max temperature data
  dataM[1:Nrecords, 3] <- temp[ind + 2] / 100.00    # Min temperature data
  dataM[1:Nrecords, 4] <- temp[ind + 3] / 100.00    # Wind speed data
  dataM[1:Nrecords, 5] <- temp[ind + 4] / 10000.00  # SPH
  dataM[1:Nrecords, 6] <- temp[ind + 5] / 40.00     # SRAD
  dataM[1:Nrecords, 7] <- temp[ind + 6] / 100.00    # Rmax
  dataM[1:Nrecords, 8] <- temp[ind + 7] / 100.00    # RMin
  AllData <- cbind(ymd, dataM)
  colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", 
                         "windspeed", "SPH", "SRAD", "Rmax", "Rmin")
  close(fileCon)
  return(AllData)
}

read_binary_addmdy_4var <- function(filename, ymd) {
  Nofvariables <- 4       # number of variables or column in the forcing data file
  Nrecords <- nrow(ymd)
  ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
  fileCon <-  file(filename, "rb")
  temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables,
                  endian="little")
  dataM <- matrix(0, Nrecords, 4)
  dataM[1:Nrecords, 1] <- temp[ind] / 40.00       # precip data
  dataM[1:Nrecords, 2] <- temp[ind + 1] / 100.00  # Max temperature data
  dataM[1:Nrecords, 3] <- temp[ind + 2] / 100.00  # Min temperature data
  dataM[1:Nrecords, 4] <- temp[ind + 3] / 100.00  # Wind speed data

  AllData <- cbind(ymd, dataM)
  # calculate daily GDD  ...what? There doesn't appear to be any GDD work?
  colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", "windspeed")
  close(fileCon)
  return(AllData)
}

create_ymdvalues <- function(data_start_year, data_end_year){

  daycount_in_year <- 0
  moncount_in_year <- 0
  yearrep_in_year <- 0
  Years <- seq(data_start_year, data_end_year)
  nYears <- length(Years)
  
  for (i in 1:nYears){
    ly <- chron::leap.year(Years[i]) # lubridate::leap_year(Years[i]) lubridate is under R 3.5.3.
    if (ly == TRUE){
      days_in_mon <- c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
      } else {
        days_in_mon <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    }
    for (j in 1:12){
      daycount_in_year <- c(daycount_in_year, seq(1, days_in_mon[j]))
      moncount_in_year <- c(moncount_in_year, rep(j, days_in_mon[j]))
      yearrep_in_year <- c(yearrep_in_year, rep(Years[i], days_in_mon[j]))
    }
  }

  daycount_in_year <- daycount_in_year[-1] # delete the leading 0
  moncount_in_year <- moncount_in_year[-1]
  yearrep_in_year <- yearrep_in_year[-1]
  ymd <- cbind(yearrep_in_year, moncount_in_year, daycount_in_year)
  colnames(ymd) <- c("year", "month", "day")
  return(ymd)
}



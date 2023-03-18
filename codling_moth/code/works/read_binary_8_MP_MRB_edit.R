# This file was originally provided to Matt Brousil (MRB) by Matt Pruett
# on 2018-09-17. Its purpose is to convert binary data files to RDS.

# This version has been edited by MRB for testing and trialing purposes.


library(lubridate)


# Define function for reading binary files --------------------------------

read_binary_8 <- function(file_name, file_path, hist){
  
  # MRB:It doesn't appear that file_name is used...?
  
  if (hist) {
    start_year <- 1979
    end_year <- 2016
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
    
    for(i in 1:nYears){
      ly <- leap_year(Years[i])
      
      if(ly == TRUE){
        days_in_mon <- c(31,29,31,30,31,30,31,31,30,31,30,31)
      }
      
      else{
        days_in_mon <- c(31,28,31,30,31,30,31,31,30,31,30,31)
      }
      
      for( j in 1:12){
        daycount_in_year <- c(daycount_in_year,seq(1,days_in_mon[j]))
        moncount_in_year <- c(moncount_in_year,rep(j,days_in_mon[j]))
        yearrep_in_year <- c(yearrep_in_year,rep(Years[i],days_in_mon[j]))
      }
    }
    
    daycount_in_year <- daycount_in_year[-1] #delete the leading 0
    moncount_in_year <- moncount_in_year[-1]
    yearrep_in_year <- yearrep_in_year[-1]
    ymd <- cbind(yearrep_in_year, moncount_in_year, daycount_in_year)
    colnames(ymd) <- c("year","month","day")
    return(ymd)
  }
  
  ## read binary data and add dates ##
  readbinarydata_addmdy <- function(filename, ymd) {
    Nofvariables <- 8 # number of variables or column in the forcing data file
    Nrecords <- nrow(ymd)
    ind <- seq(1, Nrecords*Nofvariables, Nofvariables)
    fileCon = file(filename, "rb")
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables,
                    endian="little")
    dataM <- matrix(0, Nrecords, 8)
    k <- 1
    dataM[1:Nrecords,1] <- temp[ind]/40.00  #precip data
    dataM[1:Nrecords,2] <- temp[ind+1]/100.00  #Max temperature data
    dataM[1:Nrecords,3] <- temp[ind+2]/100.00  #Min temperature data
    dataM[1:Nrecords,4] <- temp[ind+3]/100.00  #Wind speed data
    dataM[1:Nrecords,5] <- temp[ind+4]/10000.00  #SPH
    dataM[1:Nrecords,6] <- temp[ind+5]/40.00  #SRAD
    dataM[1:Nrecords,7] <- temp[ind+6]/100.00  #Rmax
    dataM[1:Nrecords,8] <- temp[ind+7]/100.00  #RMin
    AllData <- cbind(ymd, dataM)
    # calculate daily GDD  ...what? There doesn't appear to be any GDD work?
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", 
                           "winspeed", "SPH", "SRAD", "Rmax", "Rmin")
    close(fileCon)
    return(AllData)
  }
  
  # create date ranges #
  ymd_file <- create_ymdvalues(start_year, end_year)
  
  # read data #
  
  readbinarydata_addmdy(file_path, ymd_file)
  
}


# Test the functions and export -------------------------------------------


hist <- read_binary_8(file_path = file.path("test-data",
                                            "historical",
                                            "data_47.15625_-122.53125"),
                      hist = TRUE)

future <- read_binary_8(file_path = file.path("test-data",
                                              "future",
                                              "data_49.09375_-122.71875"),
                        hist = FALSE)

write.table(x = hist, file = "historical_binary_test.txt")
saveRDS(object = hist, file = "historical_binary_test.rds")
write.table(x = future, file = "future_binary_test.txt")
saveRDS(object = future, file = "future_binary_test.rds")

# Then can use readRDS() to open the .rds files...
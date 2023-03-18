library(dplyr)
library(readr)
library(lubridate)
library(parallel)

read_binary <- function(file_name, file_path){
  
  full_path <- paste0(file_path, file_name)
  
  start_year <- 1979
  end_year <- 2016
  
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
      
      for(j in 1:12){
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
    Nofvariables <- 8 # number of varaibles or column in the forcing data file
    Nrecords <- nrow(ymd)
    ind <- seq(1, Nrecords*Nofvariables, Nofvariables)
    fileCon = file(filename, "rb")
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables, endian="little")
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
    # calculate daily GDD
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", 
                           "winspeed", "SPH", "SRAD", "Rmax", "Rmin")
    close(fileCon)
    return(AllData)
  }
  
  # create date ranges #
  ymd_file <- create_ymdvalues(start_year, end_year)
  
  # read data #
  
  hist <- readbinarydata_addmdy(full_path, ymd_file) %>% 
    as_tibble() %>%
    mutate(group = "hist",
           model = "historical",
           climate_proj = NA)
  
  write_rds(hist, paste0("/data/hydro/users/mpruett/historical/", file_name, ".rds"))
  
  return(NULL)
  
}

file_path = "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"

df <- read_tsv("file_list.txt", col_names = "file_name")

mclapply(df$file_name, read_binary, file_path = file_path, mc.cores = 64)




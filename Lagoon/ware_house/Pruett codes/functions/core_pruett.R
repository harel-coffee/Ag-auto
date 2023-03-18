##############################
## storm_calc
## Calculate Gumball EV Distrobution

require(dplyr)
require(purrr)
require(tidyr)
require(tibble)

storm_calc <- function(df, percentile) {
  calc_KT <- function(return_period) {
    (-sqrt(6)/pi)*(0.5772 + log(log(return_period/(return_period - 1))))
  }
  
  calc_XT <- function(KT, mean_precip, sd_precip) {
    mean_precip + KT*sd_precip
  }
  
  df %>% as.data.frame() %>% 
    group_by(year, group) %>%
    summarise(max_hourly_precip = quantile(precip, percentile) / 24) %>%
    group_by(group) %>%
    summarise(
      mean_precip = mean(max_hourly_precip),
      sd_precip = sd(max_hourly_precip)) %>%
    mutate(
      return_period = list(seq(5, 25, 1)),
      KT = map(return_period, calc_KT),
      XT = pmap(list(KT, mean_precip, sd_precip), calc_XT)) %>%
    unnest()
}
############## 
############## water_year_tools
##############
library(lubridate)

wtr_yr <- function(dates, start_month = 10) {
  # Convert dates into POSIXlt
  dates.posix <- as.POSIXlt(dates)
  # Year offset
  offset <- ifelse(dates.posix$mon >= start_month - 1, 1, 0)
  # Water year
  adj.year <- dates.posix$year + 1900 + offset
  # Return the water year
  adj.year
}

wtr_week <- function(dates, start_month = 10) {
  offset <- ifelse(month(dates) >= start_month, 0, 1)
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]]%/%7 + 1
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]]%/%7 + 1
}

wtr_doy <- function(dates, start_month = 10) {
  offset <- ifelse(month(dates) >= start_month, 0, 1)
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]] + 1
}
##*******************************************

############## 
############## read_RDS
##############
### Load RDS Files ###

read_RDS <- function(file_name, climate_proj, model){
  hist <- readRDS(paste0("data/pruett/RDS/historical/", file_name, ".rds"))
  df_futr <- readRDS(paste0("data/pruett/RDS/",model, "/", climate_proj, "/", file_name, ".rds")) %>% 
    filter(year >= 2016)
  
  df_NA <- df_futr %>% filter(year <= 2025 || year >= 2095) %>%
    mutate(group = NA)

  df_2040 <- df_futr %>% filter(year >= 2025, year <= 2055) %>%
    mutate(group = "2040s")

  df_2060 <- df_futr %>% filter(year >= 2045, year <= 2075) %>%
    mutate(group = "2060s")

  df_2080 <- df_futr %>% filter(year >= 2065, year <= 2095) %>%
    mutate(group = "2080s")

  futr <- bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
    mutate(group = as.factor(group))

  df <- rbind(hist, futr)
  df <- mutate(df, time_stamp = ymd(paste(year, month, day, sep="-")),
           water_year = year(time_stamp + month(3)))
  return(df)
}
##*******************************************

## read binary data and add dates ##
readbinarydata_addmdy <- function(filename, Nofvariables, ymd) {
  Nrecords <- nrow(ymd)
  ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
  fileCon = file(filename, "rb")
  temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables, endian="little")
  dataM <- matrix(0, Nrecords, Nofvariables)
  close(fileCon)

  if (Nofvariables==4){
    dataM[1:Nrecords,1] <- temp[ind]/40.00    # precip data
    dataM[1:Nrecords,2] <- temp[ind+1]/100.00 # Max temperature data
    dataM[1:Nrecords,3] <- temp[ind+2]/100.00 # Min temperature data
    dataM[1:Nrecords,4] <- temp[ind+3]/100.00 # Wind speed data

    AllData <- cbind(ymd, dataM)
    # calculate daily GDD
    colnames(AllData) <- c("year", "month", "day", 
                           "precip", "tmax", "tmin", "winspeed")
   } else if (Nofvariables==8){
    dataM[1:Nrecords,1] <- temp[ind]/40.00    # precip data
    dataM[1:Nrecords,2] <- temp[ind+1]/100.00 # Max temperature data
    dataM[1:Nrecords,3] <- temp[ind+2]/100.00 # Min temperature data
    dataM[1:Nrecords,4] <- temp[ind+3]/100.00 # Wind speed data
    dataM[1:Nrecords,5] <- temp[ind+4]/10000.00  # SPH
    dataM[1:Nrecords,6] <- temp[ind+5]/40.00  # SRAD
    dataM[1:Nrecords,7] <- temp[ind+6]/100.00 # Rmax
    dataM[1:Nrecords,8] <- temp[ind+7]/100.00 # RMin

    AllData <- cbind(ymd, dataM)
    # calculate daily GDD
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", 
                           "winspeed", "SPH", "SRAD", "Rmax", "Rmin")
  }
  return(AllData)
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




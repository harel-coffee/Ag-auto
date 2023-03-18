###################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
###################################################################
options(digits=9)
read_binary <- function(filename, hist, no_vars){
  ######    The modeled historical is in /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/
  ######    modeled historical is equivalent to having 4 variables, and years 1950-2005
  ######
  ######    The observed historical is in 
  ######    /data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016
  ######    observed historical is equivalent to having 8 variables, and years 1979-2016
  ######

  # sanitation check! 

  if (hist){
    if (no_vars==4){
      start_year <- 1950
      end_year <- 2005
      print("-------------------------------")
      print (paste0("no_vars= "))
      print (no_vars)
      print("-------------------------------")
      print ("hist = ")
      print (hist)
      print("-------------------------------")
      print (start_year)
      print (end_year)
      } else {
        start_year <- 1979
        end_year <- 2016
        print("-------------------------------")
        print (paste0("no_vars= "))
        print (no_vars)
        print("-------------------------------")
        print ("hist = ")
        print (hist)
        print("-------------------------------")
        print (start_year)
        print (end_year)
      }
    } else {
      start_year <- 2006
      end_year <- 2099
      print ("file_path")
      print (file_path)
      print("-------------------------------")
      print (paste0("no_vars= "))
      print (no_vars)
      print("-------------------------------")
      print ("hist = ")
      print (hist)
      print("-------------------------------")
      print (start_year)
      print (end_year)
  }
  ymd_file <- create_ymdvalues(start_year, end_year)
  data <- read_binary_addmdy(filename, ymd_file, no_vars)
  return(data)
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
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables,
                    endian = "little")
    dataM <- matrix(0, Nrecords, 8)
    k <- 1
    dataM[1:Nrecords, 1] <- temp[ind] / 40.00         # precip data
    dataM[1:Nrecords, 2] <- temp[ind + 1] / 100.00    # Max temperature data
    dataM[1:Nrecords, 3] <- temp[ind + 2] / 100.00    # Min temperature data
    dataM[1:Nrecords, 4] <- temp[ind + 3] / 100.00    # Wind speed data
    dataM[1:Nrecords, 5] <- temp[ind + 4] / 10000.00  # SPH
    dataM[1:Nrecords, 6] <- temp[ind + 5] / 40.00     # SRAD
    dataM[1:Nrecords, 7] <- temp[ind + 6] / 100.00    # Rmax
    dataM[1:Nrecords, 8] <- temp[ind + 7] / 100.00    # RMin
    AllData <- cbind(ymd, dataM)
    # calculate daily GDD  ...what? There doesn't appear to be any GDD work?
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin",
                           "windspeed", "SPH", "SRAD", "Rmax", "Rmin")
    close(fileCon)
    return(AllData)
}

read_binary_addmdy_4var <- function(filename, ymd) {
    Nofvariables <- 4 # number of variables or column in the forcing data file
    Nrecords <- nrow(ymd)
    ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
    fileCon <-  file(filename, "rb")
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables,
                    endian="little")
    dataM <- matrix(0, Nrecords, 4)
    k <- 1
    dataM[1:Nrecords, 1] <- temp[ind] / 40.00       # precip data
    dataM[1:Nrecords, 2] <- temp[ind + 1] / 100.00  # Max temperature data
    dataM[1:Nrecords, 3] <- temp[ind + 2] / 100.00  # Min temperature data
    dataM[1:Nrecords, 4] <- temp[ind + 3] / 100.00  # Wind speed data

    AllData <- cbind(ymd, dataM)
    # calculate daily GDD  ...what? There doesn't appear to be any GDD work?
    colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin",
                           "windspeed")
    close(fileCon)
    return(AllData)
}

create_ymdvalues <- function(data_start_year, data_end_year){
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
######################################################################
######################################################################
######################################################################
######################################################################

# Function to compute and append gdd and cumulative gdd columns to met data
add_dd_cumdd <- function(metdata_data.table, lower=10, upper=31.11) {
  twopi  = 2*pi
  pihalf = pi/2
  diff = metdata_data.table$tmax - metdata_data.table$tmin # column diff
  tsum = metdata_data.table$tmax + metdata_data.table$tmin # column tsum
  
  aveminlt = (tsum/2) - lower
  alpha1 = diff / 2
  avetemp = tsum / 2
  
  theta1 = asin((lower - avetemp) / alpha1)
  theta2 = asin((upper - avetemp) / alpha1)
  
  tmin = metdata_data.table$tmin
  tmax = metdata_data.table$tmax
  
  tempdata = data.frame(tmin, tmax, diff, tsum, aveminlt, alpha1, avetemp, theta1, theta2)
  tempdata$heat = 0

  #case 1 tmin >= upper threshold
  tempdata$heat[tempdata$tmin >= upper] = upper - lower
  
  #case 2 is handled by the default where heat is set to 0 initially it is when tmax<lower threshold
  
  #case 3 tmin>= lower threshold and the max <= upper threshold
  tempdata$heat[tempdata$tmin >= lower & tempdata$tmax <= upper] = tempdata$aveminlt[tempdata$tmin >= lower & tempdata$tmax <= upper]
  
  #case 4 tmin<lower threshold and tmax>lower and tmax<=upper threshold
  tempdata$heat[tempdata$tmin < lower & 
                tempdata$tmax > lower & tempdata$tmax <= upper] = (((tempdata$aveminlt[tempdata$tmin < lower & 
                                                                    tempdata$tmax > lower & 
                                                                    tempdata$tmax <= upper] * 
                                                                    (pihalf - 
                                                                    tempdata$theta1[tempdata$tmin < lower & 
                                                                    tempdata$tmax > lower & 
                                                                    tempdata$tmax <= upper])) + 
                                                                    (tempdata$alpha1[tempdata$tmin < lower & 
                                                                    tempdata$tmax > lower & 
                                                                    tempdata$tmax <= upper] 
                                                                    * cos(tempdata$theta1[tempdata$tmin < lower & 
                                                                    tempdata$tmax > lower & tempdata$tmax <= upper]))) / pi)
  
  #case 5 tmin>=lower threshold & tmin<upper & tmax>upper threshold
  tempdata$heat[tempdata$tmin >= lower & tempdata$tmin < upper & 
                tempdata$tmax > upper] = (((tempdata$aveminlt[tempdata$tmin >= lower & 
                                            tempdata$tmin < upper & tempdata$tmax > upper] * 
                                            (tempdata$theta2[tempdata$tmin >= lower & 
                                            tempdata$tmin < upper & tempdata$tmax > upper] + 
                                            pihalf)) + ((upper - lower) * 
                                            (pihalf - tempdata$theta2[tempdata$tmin >= lower & 
                                            tempdata$tmin < upper & tempdata$tmax > upper])) - 
                                            (tempdata$alpha1[tempdata$tmin >= lower & 
                                            tempdata$tmin < upper & tempdata$tmax > upper] * 
                                            cos(tempdata$theta2[tempdata$tmin >= lower & 
                                              tempdata$tmin < upper & tempdata$tmax > upper]))) / pi)
 
  #case 6 tmin<lower threshold & tmax>upper threshold
  tempdata$heat[tempdata$tmin < lower & tempdata$tmax > upper] = (((tempdata$aveminlt[tempdata$tmin < 
                                                                    lower & tempdata$tmax > upper] * 
                                                                    (tempdata$theta2[tempdata$tmin < lower & 
                                                                    tempdata$tmax > upper] - 
                                                                    tempdata$theta1[tempdata$tmin < 
                                                                    lower & tempdata$tmax > upper])) + 
                                                                    (tempdata$alpha1[tempdata$tmin < lower & 
                                                                    tempdata$tmax > upper] * 
                                                                    (cos(tempdata$theta1[tempdata$tmin < lower & 
                                                                    tempdata$tmax > upper]) - 
                                                                    cos(tempdata$theta2[tempdata$tmin < lower & 
                                                                    tempdata$tmax > upper]))) + 
                                                                    ((upper - lower) * (pihalf - 
                                                                    tempdata$theta2[tempdata$tmin < lower & 
                                                                    tempdata$tmax > upper]))) / pi)
  
  metdata_data.table$dd <- tempdata$heat
  metdata_data.table[, Cum_dd := cumsum(dd), by=list(year)]
  return(metdata_data.table)
}

merge_add_countyGroup <- function(input_dir, param_dir, 
                                  locations_file_name,
                                  locationGroup_fileName,
                                  categories, file_prefix, version){
  merged_data <- merge_data(input_dir, param_dir, categories, locations_file_name, file_prefix, version)
  merged_data = add_countyGroup(merged_data, param_dir, loc_group_file_name= locationGroup_fileName)
  return(merged_data)
}

add_countyGroup <- function(data, param_dir, loc_group_file_name){
  loc_grp = data.table(read.csv(paste0(param_dir, loc_group_file_name)))
  options(digits=9)
  loc_grp$latitude = as.numeric(loc_grp$latitude)
  loc_grp$longitude = as.numeric(loc_grp$longitude)

  data$CountyGroup = 0L

  for(i in 1:nrow(loc_grp)) {
    data[latitude == loc_grp[i, latitude] & longitude == loc_grp[i, longitude], ]$CountyGroup = loc_grp[i, locationGroup]
  }
  return (data) 
}

merge_data <- function(input_dir, param_dir, categories, locations_file_name, file_prefix, version){
  data = data.table()
  
  conn = file(paste0(param_dir, locations_file_name), open = "r")
  locations = readLines(conn)
  close(conn)

  for(category in categories){
    for(location in locations){
      if(category != "historical") {
        filename <- paste0(input_dir, "future_", file_prefix, "/", category, "/", version, "/", file_prefix, "_", location)
      }
      else {
        filename <- paste0(input_dir, "historical_", file_prefix, "/", file_prefix, "_" ,location)
      }
    data_to_add = read.table(filename, header = TRUE, sep = ",")
    data <- rbind(data, data_to_add)
    }
  }
  return(data)
}

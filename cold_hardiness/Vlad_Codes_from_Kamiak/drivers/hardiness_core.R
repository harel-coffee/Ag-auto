
options(digits=9)
options(digit=9)
####################################################################
##                                                                ##
##                                                                ##
##                            Analysis                            ##
##                                                                ##
##                                                                ##
####################################################################

read_binary <- function(file_path, hist, no_vars){
  ######    The modeled historical is in /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/
  ######    modeled historical is equivalent to having 4 variables, and years 1950-2005
  ######
  ######    The observed historical is in 
  ######    /data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016
  ######    observed historical is equivalent to having 8 variables, and years 1979-2016
  ######
  if (hist) {
    if (no_vars==4){
      start_year <- 1950
      end_year <- 2005
    } else {
      start_year <- 1979
      end_year <- 2015
    }
  } else{
    start_year <- 2006
    end_year <- 2099
  }
  ymd_file <- create_ymdvalues(start_year, end_year)
  data <- read_binary_addmdy(file_path, ymd_file, no_vars)
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


hardiness_model <- function(data, 
                            input_params = input_params,
                            variety_params = variety_params){
  options(digits=9)
  output = data.table(matrix(NA, nrow=dim(data)[1], ncol=14))
  colnames(output) <- c(
    "variety","year", "Date", "jday", "t_mean", "t_max", "t_min", "predicted_Hc",
    "budbreak", "predicted_on", "CDI", 'dormancy_period', 'DD_heating_sum', 'base10_chilling_sum')
  
  #output$predicted_on <- as.character(output$predicted_on)
  
  output$variety <- as.character(output$variety)
  #output$location <- as.character(output$location)
  
  #location = data$Location[1]
  # year_1 = data$year
  # The following is just name of a given column
  # to be used to name a column in the output sheet/file.
  temp <- colnames(data)[7]
  variety <- input_params$variety[1]
  
  Hc_initial <- input_params$Hc_initial[1]
  Hc_min<- input_params$Hc_min[1]
  Hc_max <- input_params$Hc_max[1]
  ####################################################################
  #################
  ################# In the following vectors of size 2, the first
  ################# entry is _endo and the second is _eco type.
  #################
  ####################################################################
  T_threshold <- c(input_params$t_threshold_endo[1], 
                   input_params$t_threshold_eco[1])
  # print(T_threshold)
  # print(input_params$t_threshold_endo[1])
  # print(input_params$t_threshold)
  
  acclimation_rate <- c(input_params$acclimation_rate_endo[1],
                        input_params$acclimation_rate_eco[1])
  
  
  deacclimation_rate<- c(input_params$deacclimation_rate_endo[1],
                         input_params$deacclimation_rate_eco[1])
  ####################################################################
  ecodormancy_boundary <- input_params$Ecodormancy_boundary[1]
  theta <- c(1, input_params$theta[1])
  
  # calculate range of hardiness values possible, 
  # this is needed for the logistic component
  ################################################## What the hell is going on here?
  Hc_range = Hc_min - Hc_max 
  
  # initialize some of the parameters
  DD_heating_sum = 0
  DD_chilling_sum = 0
  base10_chilling_sum = 0
  model_Hc_yesterday = Hc_initial
  dormancy_period = 1
  
  # number of observations 
  n_rows = dim(data)[1]
  
  #changing format of the Date column to Date format
  data$Date <- as.Date(data$Date, format ="%m/%d/%Y")
  
  
  for (row_count in 1:n_rows){
    # print(row_count)
    # jdate <- data[row_count, 1]
    # the following line is done so we can write
    # the result in CSV format.
    # (when class of the variable was of format factor, it had problem)
    jdate = as.character((data$Date[row_count]))
    jday = data$jday[row_count]   # jday <- data[row_count, 3]
    # t_mean= data$T_mean[row_count] # t_mean = data[row_count, 5]
    # print(T_threshold[dormancy_period])
    
    # print(sapply(data$Date, class))
    
    
    if(format(data$Date[row_count], "%m")=="09" && format(data$Date[row_count], "%d")=="01")
    # if (grepl("9/1",meta_data$Date[row_count])== TRUE)
    {
      Hc_initial <- input_params$Hc_initial[1]
      Hc_min<- input_params$Hc_min[1]
      Hc_max <- input_params$Hc_max[1] 
      
      
      T_threshold <- c(input_params$t_threshold_endo[1], 
                       input_params$t_threshold_eco[1])
      # print(T_threshold)
      # print(input_params$t_threshold_endo[1])
      # print(input_params$t_threshold)
      
      acclimation_rate <- c(input_params$acclimation_rate_endo[1],
                            input_params$acclimation_rate_eco[1])
      
      
      deacclimation_rate<- c(input_params$deacclimation_rate_endo[1],
                             input_params$deacclimation_rate_eco[1])
      ####################################################################
      ecodormancy_boundary <- input_params$Ecodormancy_boundary[1]
      theta <- c(1, input_params$theta[1])
      
      
      DD_heating_sum = 0
      DD_chilling_sum = 0
      base10_chilling_sum = 0
      model_Hc_yesterday = Hc_initial
      dormancy_period = 1
    }
    

    if (is.na(data$T_mean[row_count])){
      message(sprintf("data$T_mean[row_count] is empty (NA) at row %s\n", row_count))
      break
    }
    t_max = data$tmax[row_count]    # t_max = data[row_count, 6]
    t_min = data$tmin[row_count]    # t_min = data[row_count, 7]
    # observed_Hc = data$Observed_Hc[row_count] # observed_Hc = data[row_count, 8]

    # calculate heating degree days for today used in deacclimation
    if (data$T_mean[row_count] > T_threshold[dormancy_period]){
      DD_heating_today <- data$T_mean[row_count] - T_threshold[dormancy_period]
    } else {
      DD_heating_today = 0
    }

    # calculate cooling degree days for today used in acclimation
    if (data$T_mean[row_count] <= T_threshold[dormancy_period]){
      DD_chilling_today = data$T_mean[row_count] - T_threshold[dormancy_period]
    } else{
      DD_chilling_today = 0
    }

    # calculate cooling degree days using base
    # of 10c to be used in dormancy release
    if(data$T_mean[row_count] <= 10){
      base10_chilling_today = data$T_mean[row_count] - 10
    } else {
      base10_chilling_today = 0
    }

    # calculate new model_Hc for today
    deacclimation = DD_heating_today * deacclimation_rate[dormancy_period] *
      (1 - ((model_Hc_yesterday - Hc_max) / Hc_range) ^ theta[[dormancy_period]])

    # do not allow deacclimation unless
    # some chilling has occured,
    # the actual start of the model
    if (DD_chilling_sum == 0){ deacclimation = 0}

    acclimation = DD_chilling_today * acclimation_rate[dormancy_period] *
      (1 - (Hc_min - model_Hc_yesterday) / Hc_range)
    Delta_Hc = acclimation + deacclimation
    model_Hc = model_Hc_yesterday + Delta_Hc

    # limit the hardiness to known min and max
    if (model_Hc <= Hc_max) {model_Hc = Hc_max}
    if (model_Hc > Hc_min) { model_Hc = Hc_min }

    # sum up chilling degree days
    DD_chilling_sum = DD_chilling_sum + DD_chilling_today

    base10_chilling_sum = base10_chilling_sum + base10_chilling_today

    # sum up heating degree days only if chilling requirement has been met
    #  i.e dormancy period 2 has started
    if (dormancy_period == 2) {DD_heating_sum = DD_heating_sum + DD_heating_today}

    output$variety[row_count] = as.character(variety)
#    output$location[row_count] = as.character(location)
    output$year[row_count] = data$year[row_count]
    output$Date[row_count] = jdate
    output$jday[row_count] = jday
    output$t_mean[row_count] = data$T_mean[row_count]
    output$t_max[row_count] = t_max
    output$t_min[row_count] = t_min

    output$predicted_Hc[row_count] = model_Hc
    # output$observed_Hc[row_count] = observed_Hc
    output$CDI[row_count] = if(output$t_min[row_count] < output$predicted_Hc[row_count]) 1 else 0

    output$dormancy_period[row_count] <- dormancy_period
    output$DD_heating_sum[row_count] <- DD_heating_sum
    output$base10_chilling_sum[row_count] <- base10_chilling_sum
    
    # determine if chilling requirement has been met
    # re-set dormancy period
    # order of this and other if statements
    # are consistant with Ferguson et al, or V6.3 of our SAS code
    if (base10_chilling_sum <= ecodormancy_boundary){dormancy_period = 2}
    
    # use Hc_min to determine if vinifera or labrusca
    if (Hc_min == -1.2){
      if(model_Hc_yesterday < -2.2){
        if(model_Hc >= -2.2){
          output$predicted_on[1] = jdate
          output$budbreak[row_count] = model_Hc
        }
      }
    }

    if(Hc_min == -2.5){
      if(model_Hc_yesterday < -6.4){
        if(model_Hc >= -6.4){
          output$predicted_on[1] = jdate
          output$budbreak[row_count] = model_Hc
        }
      }
    }

    # remember todays hardiness for tomarrow
    model_Hc_yesterday = model_Hc
  }
  # output$CDI = if(output$t_min < output$predicted_Hc) 1 else 0 
  # output <- output %>% mutate(CDI = if(output$t_min < output$predicted_Hc) 1 else 0 )
  return(output)
}


  
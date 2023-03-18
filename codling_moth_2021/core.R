options(digits=9)

source_path = "/home/hnoorazar/reading_binary/read_binary_core.R"
source(source_path)
#####################################################################
#############                                                   #####
#############  Phase 1: Read binary data and parameters and     #####
#############           generate CM and CMPOP files.            #####
#############                                                   #####
#####################################################################
produce_CMPOP_local <- function (input_folder, noVariables, filename, 
                                 param_dir, cod_moth_param_name, 
                                 scale_shift = 0,
                                 start_year, end_year,
                                 lower=10, upper=31.11,
                                 location, category){

  # The name 'local' in the function name has no meaning.
  # It has remained there from the past! 

  temp <- prepareData_CMPOP(filename= filename, 
                            input_folder = input_folder, 
                            param_dir = param_dir, 
                            cod_moth_param_name = cod_moth_param_name,
                            scale_shift = 0,
                            start_year = start_year, end_year = end_year, 
                            NoVariables = noVariables,
                            lower = lower, upper = upper)
  
  if (category== "observed"){
    temp <- temp %>% filter(year >= start_year & year <= end_year) %>% data.table()
    temp$time_period <- paste0(start_year, "-", end_year)
    
  } else {
    data_F1 <- temp %>% filter(year >= 2025 & year <= 2055) %>% data.table()
    data_F1$time_period <- "2040s"

    data_F2 <- temp %>% filter(year >= 2045 & year <= 2075) %>% data.table()
    data_F2$time_period <- "2060s"

    data_F3 <- temp %>% filter(year >= 2065 & year <= 2095) %>% data.table()
    data_F3$time_period <- "2080s"

    temp <- rbind(data_F1, data_F2, data_F3)
  }

  loc = tstrsplit(location, "_")
  options(digits=9)
  temp$latitude <- as.numeric(unlist(loc[1]))
  temp$longitude <- as.numeric(unlist(loc[2]))
  temp$County <- as.character(unique(cellByCounty[lat == temp$latitude[1] & 
                                                  long == temp$longitude[1], 
                                                  countyname]))
  temp$model <- category
  return (temp)
}

#####################################################################
produce_CM_local <- function(input_folder, NoVariables, filename,
                             param_dir, cod_moth_param_name,
                             scale_shift = 0,
                             start_year, end_year, 
                             lower = 10, upper = 31.11,
                             location, category){

  loc = tstrsplit(location, "_")

  temp <- prepareData_CM(filename = filename, 
                         input_folder = input_folder, 
                         param_dir = param_dir, 
                         cod_moth_param_name = cod_moth_param_name,
                         scale_shift = scale_shift,
                         start_year = start_year, end_year = end_year,
                         NoVariables = NoVariables, 
                         lower = lower, upper = upper)

  if (category == "observed"){
    temp <- temp %>% filter(year >= start_year & year <= end_year) %>% data.table()
    temp$time_period <- paste0(start_year, "-", end_year)
    } 
  else{

    data_F1 <- temp %>% filter(year >= 2025 & year <= 2055) %>% data.table()
    data_F1$time_period <- "2040s"

    data_F2 <- temp %>% filter(year >= 2045 & year <= 2075) %>% data.table()
    data_F2$time_period <- "2060s"

    data_F3 <- temp %>% filter(year >= 2065 & year <= 2095) %>% data.table()
    data_F3$time_period <- "2080s"

    temp <- rbind(data_F1, data_F2, data_F3)
  }
  options(digits=9)
  temp$latitude <- as.numeric(unlist(loc[1]))
  temp$longitude <- as.numeric(unlist(loc[2]))
  temp$County <- as.character(unique(cellByCounty[lat == temp$latitude[1] & 
                                                  long == temp$longitude[1], 
                                                   countyname]))
  temp$model <- category
  return (temp)
}
############################################
###############
###############   prepareData_CMPOP   ######
###############
############################################
prepareData_CMPOP <- function(filename, input_folder,
                              param_dir, cod_moth_param_name,
                              scale_shift = 0,
                              start_year, end_year, NoVariables,
                              lower, upper){
  
  # time_stuff <- provide_time_stuff(start_year = start_year, end_year = end_year, Nofvariables=4)
  # nYears <- time_stuff[[1]]
  # Nrecords <- time_stuff[[2]]
  # Nofvariables <- time_stuff[[3]]
  # Years <- time_stuff[[4]]
  # ind <- time_stuff[[5]]
  
  # create year, month, day values based on start year, number of years and leap year info
  # ymd <- create_ymdvalues (nYears, Years, leap.year)
  A_ymd <- create_ymdvalues (data_start_year = start_year , data_end_year = end_year )
  
  ##
  ## read met data and add year, month, day variables
  ##
  input_file <- paste(input_folder, filename, sep="")

  # metdata <- readbinarydata_addmdy(input_file, Nrecords, Nofvariables, ymd, ind)

  # from read_binanry_core.R
  metdata <- read_binary_addmdy(filename=input_file, ymd=A_ymd, no_vars=NoVariables) 
  metdata <- data.table(metdata)

  
  # Calculate daily and cumulative gdd to met data. (gdd := growing degree days)
  # why three is no 32 in conversion?
  metdata <- add_dd_cumdd(metdata, lower, upper)
  metdata$Cum_dd_F = metdata$Cum_dd * 1.8

  # add day of year from 1 to 365/366 depending on year
  metdata$dum <- 1 # dummy
  metdata[, dayofyear := cumsum(dum), by=list(year)] #?
  
  CodMothParams <- read.table(paste0(param_dir, cod_moth_param_name), header=TRUE, sep=",")
  
  # Generate Relative Population
  relpopulation <- CodlingMothRelPopulation(CodMothParams, metdata, scale_shift)
  
  data <- cbind(metdata$tmax, metdata$tmin, 
                metdata$dd, metdata$Cum_dd, 
                metdata$Cum_dd_F, relpopulation, 
                metdata$day )

  rel_col_names <- colnames(relpopulation)[1:8]
  colnames(data) <- c("tmax", "tmin", "DailyDD", "CumDDinC", "CumDDinF", 
                      rel_col_names, 
                      "SumEgg", "SumLarva", "SumPupa",
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  rm (relpopulation)
  # Generate Percent Population
  percpopulation <- CodlingMothPercentPopulation(CodMothParams, metdata, scale_shift)
  
  data <- cbind(percpopulation[, 1:12], data)
  colnames(data) <- c(colnames(percpopulation)[1:8], "PercEgg", 
                      "PercLarva", "PercPupa", "PercAdult", 
                      "tmax", "tmin", "DailyDD", "CumDDinC", 
                      "CumDDinF", rel_col_names, 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  rm (percpopulation)
  # reorder the dataframe.
  # (why? probably because the way the rest of 
  # the code were written, before writing this function)
  data <- data[, c(31:33, 30, 13:17, 18:29, 1:12)]
  return(data)
}

########################################################################################################
########################################################################################################
prepareData_CM <- function(filename, input_folder, 
                           param_dir, cod_moth_param_name, 
                           scale_shift = 0,
                           start_year, end_year, 
                           NoVariables,
                           lower, upper){
  # time_stuff <- provide_time_stuff(start_year, end_year)
  # nYears <- time_stuff[[1]]
  # Nrecords <- time_stuff[[2]]
  # Nofvariables <- time_stuff[[3]]
  # Years <- time_stuff[[4]]
  # ind <- time_stuff[[5]]
  
  ## create year, month, day values based on start year, number of years and leap year info
  # ymd <- create_ymdvalues(nYears, Years, leap.year) # commented out on Feb 19, 2021.
  A_ymd <- create_ymdvalues (data_start_year = start_year , data_end_year = end_year )
  
  ## read met data and add year month day variables
  input_file <- paste(input_folder, filename, sep="")
  

  # metdata <- readbinarydata_addmdy(input_file, Nrecords, Nofvariables, ymd, ind)
  # from read_binanry_core.R
  metdata <- read_binary_addmdy(filename=input_file, ymd = A_ymd, no_vars = NoVariables) 
  metdata <- data.table(metdata)
  
  # calculate daily and cumulative gdd to met data
  metdata <- add_dd_cumdd(metdata, lower, upper)
  
  # convert celcius to farenheit
  metdata$Cum_dd_F = metdata$Cum_dd * 1.8 
  
  # add day of year from 1 to 365/366 depending on year
  metdata$dum <- 1 # dummy
  metdata[, dayofyear := cumsum(dum), by=list(year)]
  
  CodMothParams <- read.table(paste0(param_dir, cod_moth_param_name), header=TRUE, sep=",")
  
  # Generate Relative Population
  relpopulation <- CodlingMothRelPopulation(CodMothParams, metdata, scale_shift)
  
  data <- cbind(metdata$tmax, metdata$tmin, metdata$dd, 
                metdata$Cum_dd, metdata$Cum_dd_F, 
                relpopulation, metdata$day )
  rel_col_names = colnames(relpopulation)[1:8]
  colnames(data) <- c("tmax", "tmin", 
                      "DailyDD", "CumDDinC", "CumDDinF", 
                      rel_col_names, 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  rm (relpopulation)

  # Generate Percent Population
  percpopulation <- CodlingMothPercentPopulation(CodMothParams, metdata, scale_shift)
  rm(metdata)
  data <- cbind(percpopulation[, 1:12], data)

  prec_col_names = colnames(percpopulation)[1:8]
  colnames(data) <- c(prec_col_names, "PercEgg", 
                      "PercLarva", "PercPupa", "PercAdult", 
                      "tmax", "tmin", "DailyDD", "CumDDinC", 
                      "CumDDinF", rel_col_names, 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  rm (percpopulation)
  data <- data[, c(31:33, 30, 13:17, 18:29, 1:12)]
  
  # for each year
  generations <- data.frame("year" = integer(), 
                            AGen1_0.25 = integer(), AGen2_0.25 = integer(), AGen3_0.25 = integer(), AGen4_0.25 = integer(),
                            AGen1_0.5  = integer(), AGen2_0.5  = integer(), AGen3_0.5  = integer(), AGen4_0.5  = integer(),  
                            AGen1_0.75 = integer(), AGen2_0.75 = integer(), AGen3_0.75 = integer(), AGen4_0.75 = integer(),
                            
                            LGen1_0.25 = integer(), LGen2_0.25 = integer(), LGen3_0.25 = integer(), LGen4_0.25 = integer(),
                            LGen1_0.5  = integer(), LGen2_0.5  = integer(), LGen3_0.5  = integer(), LGen4_0.5  = integer(),  
                            LGen1_0.75 = integer(), LGen2_0.75 = integer(), LGen3_0.75 = integer(), LGen4_0.75 = integer(),
                            
                            Emergence = integer(),
                            
                            AdultFeb1 =numeric(), AdultMar1 =numeric(), AdultApr1 =numeric(), AdultMay1 =numeric(), 
                            AdultJune1 =numeric(), AdultJul1 =numeric(), AdultAug1 =numeric(), AdultSep1 =numeric(),
                            AdultOct1 =numeric(), AdultNov1 =numeric(), AdultDec1 =numeric(),
                            AdultGen1Feb1 =numeric(), AdultGen1Mar1 =numeric(), AdultGen1Apr1 =numeric(), AdultGen1May1 =numeric(), 
                            AdultGen1June1 =numeric(), AdultGen1Jul1 =numeric(), AdultGen1Aug1 =numeric(), AdultGen1Sep1 =numeric(),
                            AdultGen1Oct1 =numeric(), AdultGen1Nov1 =numeric(), AdultGen1Dec1 =numeric(),
                            AdultGen2Feb1 =numeric(), AdultGen2Mar1 =numeric(), AdultGen2Apr1 =numeric(), AdultGen2May1 =numeric(), 
                            AdultGen2June1 =numeric(), AdultGen2Jul1 =numeric(), AdultGen2Aug1 =numeric(), AdultGen2Sep1 =numeric(),
                            AdultGen2Oct1 =numeric(), AdultGen2Nov1 =numeric(), AdultGen2Dec1 =numeric(),
                            AdultGen3Feb1 =numeric(), AdultGen3Mar1 =numeric(), AdultGen3Apr1 =numeric(), AdultGen3May1 =numeric(), 
                            AdultGen3June1 =numeric(), AdultGen3Jul1 =numeric(), AdultGen3Aug1 =numeric(), AdultGen3Sep1 =numeric(),
                            AdultGen3Oct1 =numeric(), AdultGen3Nov1 =numeric(), AdultGen3Dec1 =numeric(),
                            AdultGen4Feb1 =numeric(), AdultGen4Mar1 =numeric(), AdultGen4Apr1 =numeric(), AdultGen4May1 =numeric(), 
                            AdultGen4June1 =numeric(), AdultGen4Jul1 =numeric(), AdultGen4Aug1 =numeric(), AdultGen4Sep1 =numeric(),
                            AdultGen4Oct1 =numeric(), AdultGen4Nov1 =numeric(), AdultGen4Dec1 =numeric(),
                            
                            LarvaFeb1 =numeric(), LarvaMar1 =numeric(), LarvaApr1 =numeric(), LarvaMay1 =numeric(),
                            LarvaJune1 =numeric(), LarvaJul1 =numeric(), LarvaAug1 =numeric(), LarvaSep1 =numeric(), 
                            LarvaOct1 =numeric(), LarvaNov1 =numeric(), LarvaDec1 =numeric(), 
                            LarvaSep15=numeric(),
                            
                            LarvaGen1Feb1 =numeric(), LarvaGen1Mar1 =numeric(), LarvaGen1Apr1 =numeric(), LarvaGen1May1 =numeric(),
                            LarvaGen1June1 =numeric(), LarvaGen1Jul1 =numeric(), LarvaGen1Aug1 =numeric(), LarvaGen1Sep1 =numeric(), 
                            LarvaGen1Oct1 =numeric(), LarvaGen1Nov1 =numeric(), LarvaGen1Dec1 =numeric(), 
                            
                            LarvaGen2Feb1 =numeric(), LarvaGen2Mar1 =numeric(), LarvaGen2Apr1 =numeric(), LarvaGen2May1 =numeric(),
                            LarvaGen2June1 =numeric(), LarvaGen2Jul1 =numeric(), LarvaGen2Aug1 =numeric(), LarvaGen2Sep1 =numeric(), 
                            LarvaGen2Oct1 =numeric(), LarvaGen2Nov1 =numeric(), LarvaGen2Dec1 =numeric(), 
                            
                            LarvaGen3Feb1 =numeric(), LarvaGen3Mar1 =numeric(), LarvaGen3Apr1 =numeric(), LarvaGen3May1 =numeric(),
                            LarvaGen3June1 =numeric(), LarvaGen3Jul1 =numeric(), LarvaGen3Aug1 =numeric(), LarvaGen3Sep1 =numeric(), 
                            LarvaGen3Oct1 =numeric(), LarvaGen3Nov1 =numeric(), LarvaGen3Dec1 =numeric(), 
                            
                            LarvaGen4Feb1 =numeric(), LarvaGen4Mar1 =numeric(), LarvaGen4Apr1 =numeric(), LarvaGen4May1 =numeric(),
                            LarvaGen4June1 =numeric(), LarvaGen4Jul1 =numeric(), LarvaGen4Aug1 =numeric(), LarvaGen4Sep1 =numeric(), 
                            LarvaGen4Oct1 =numeric(), LarvaGen4Nov1 =numeric(), LarvaGen4Dec1 =numeric(), 
                            
                            Diapause = numeric()
  )
  
  for (i in min(data$year):max(data$year))
  {
    Agen = "AGen"
    Lgen = "LGen"

    atf = 0   # 25 occured adult
    aff = 0   # 50 occured adult
    asf = 0   # 75 occured adult
    acurr = 0 # current % population per generation Adult
    ltf = 0   # 25 occured larva
    lff = 0   # 50 occured larva
    lsf = 0   # 75 occured larva
    lcurr = 0 # current % population per generation Larva
    agen = 1  # generation of the year adults
    lgen = 1  # generation of the year for larva
    
    em = 0 # emergence occured
    
    # add each year to dataframe
    generations[nrow(generations) + 1, 1] <- i
    
    # for each day of the year
    for (j in 1:nrow(subset(data, year == i)))
    {
      # get the data for each day
      row <- subset(data, year == i)[j, ]
      aperc     <- row$PercAdult     # current % adults
      aGen1perc <- row$PercAdultGen1
      aGen2perc <- row$PercAdultGen2
      aGen3perc <- row$PercAdultGen3
      aGen4perc <- row$PercAdultGen4
      lperc     <- row$PercLarva     # current % larva
      lGen1perc <- row$PercLarvaGen1
      lGen2perc <- row$PercLarvaGen2
      lGen3perc <- row$PercLarvaGen3
      lGen4perc <- row$PercLarvaGen4

      #####################################################################################
      #######################                                            ##################
      #######################               DIAPAUSE                     ##################
      #######################                                            ##################
      #####################################################################################
      if (row$month == 8 & row$day == 13)
      {
        generations[generations$year == i,]["Diapause"] <- lperc
      }
      #####################################################################################
      #######################                                            ##################
      #######################   Percentages at the start of each month   ##################
      #######################                                            ##################
      #####################################################################################
      if (row$month == 2 & row$day == 1)
      {
        generations[generations$year == i,]["AdultFeb1"] <- aperc
        generations[generations$year == i,]["LarvaFeb1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Feb1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Feb1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Feb1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Feb1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Feb1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Feb1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Feb1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Feb1"] <- lGen4perc
      }
      
      if (row$month == 3 & row$day == 1)
      {
        generations[generations$year == i,]["AdultMar1"] <- aperc
        generations[generations$year == i,]["LarvaMar1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Mar1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Mar1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Mar1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Mar1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Mar1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Mar1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Mar1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Mar1"] <- lGen4perc
      }
      
      if (row$month == 4 & row$day == 1)
      {
        generations[generations$year == i,]["AdultApr1"] <- aperc
        generations[generations$year == i,]["LarvaApr1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Apr1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Apr1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Apr1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Apr1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Apr1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Apr1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Apr1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Apr1"] <- lGen4perc
      }
      
      if (row$month == 5 & row$day == 1)
      {
        generations[generations$year == i,]["AdultMay1"] <- aperc
        generations[generations$year == i,]["LarvaMay1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1May1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1May1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2May1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2May1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3May1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3May1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4May1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4May1"] <- lGen4perc
      }
      
      if (row$month == 6 & row$day == 1)
        
      {
        generations[generations$year == i,]["AdultJune1"] <- aperc
        generations[generations$year == i,]["LarvaJune1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1June1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1June1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2June1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2June1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3June1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3June1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4June1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4June1"] <- lGen4perc
      }
      
      if (row$month == 7 & row$day == 1)
        
      {
        generations[generations$year == i,]["AdultJul1"] <- aperc
        generations[generations$year == i,]["LarvaJul1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Jul1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Jul1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Jul1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Jul1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Jul1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Jul1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Jul1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Jul1"] <- lGen4perc
      }
      
      if (row$month == 8 & row$day == 1)
        
      {
        generations[generations$year == i,]["AdultAug1"] <- aperc
        generations[generations$year == i,]["LarvaAug1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Aug1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Aug1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Aug1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Aug1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Aug1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Aug1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Aug1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Aug1"] <- lGen4perc
      }
      
      if (row$month == 9 & row$day == 1)
        
      {
        generations[generations$year == i,]["AdultSep1"] <- aperc
        generations[generations$year == i,]["LarvaSep1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Sep1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Sep1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Sep1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Sep1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Sep1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Sep1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Sep1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Sep1"] <- lGen4perc
      }
      if (row$month == 9 & row$day == 15){
        generations[generations$year == i,]["LarvaSep15"] <- lperc
      }
      
      if (row$month == 10 & row$day == 1){
        generations[generations$year == i,]["AdultOct1"] <- aperc
        generations[generations$year == i,]["LarvaOct1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Oct1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Oct1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Oct1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Oct1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Oct1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Oct1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Oct1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Oct1"] <- lGen4perc
      }
      
      if (row$month == 11 & row$day == 1){
        generations[generations$year == i,]["AdultNov1"] <- aperc
        generations[generations$year == i,]["LarvaNov1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Nov1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Nov1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Nov1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Nov1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Nov1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Nov1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Nov1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Nov1"] <- lGen4perc
      }
      
      if (row$month == 12 & row$day == 1){
        generations[generations$year == i,]["AdultDec1"] <- aperc
        generations[generations$year == i,]["LarvaDec1"] <- lperc
        
        generations[generations$year == i,]["AdultGen1Dec1"] <- aGen1perc
        generations[generations$year == i,]["LarvaGen1Dec1"] <- lGen1perc
        
        generations[generations$year == i,]["AdultGen2Dec1"] <- aGen2perc
        generations[generations$year == i,]["LarvaGen2Dec1"] <- lGen2perc
        
        generations[generations$year == i,]["AdultGen3Dec1"] <- aGen3perc
        generations[generations$year == i,]["LarvaGen3Dec1"] <- lGen3perc
        
        generations[generations$year == i,]["AdultGen4Dec1"] <- aGen4perc
        generations[generations$year == i,]["LarvaGen4Dec1"] <- lGen4perc
      }
      #####################################################################################
      #######################                                            ##################
      #######################                 Emergence                  ##################
      #######################                                            ##################
      #####################################################################################
    
      if (row$CumDDinC > 100 & em == 0){
        em = 1
        generations[generations$year == i,]["Emergence"] <- row$dayofyear
      }
      #####################################################################################
      #######################                                            ##################
      #######################          GENERATIONS of ADULTS             ##################
      #######################                                            ##################
      #####################################################################################

      # continue to increase the percentage
      if (aperc >= acurr) { acurr = aperc }
      # if there is a reset (start of a new generation)
      else{
        agen = agen + 1 # increase generation count
        
        # now reset boolean for occuring percentages (for new generation)
        atf = 0
        aff = 0
        asf = 0
        acurr = 0
      }
      # if new gen reached 25
      if (aperc > .25 & atf == 0){
        atf = 1 # check off that this occured
        col <- paste0(Agen, agen, "_", "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      # if new gen reached 50
      if (aperc > .50 & aff == 0){
        aff = 1 #check off that this occured
        col <- paste0(Agen, agen, "_", "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      # if new gen reached 75
      if (aperc > .75 & asf == 0){
        asf = 1 #check off that this occured
        col <- paste0(Agen, agen, "_", "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      #####################################################################################
      #######################                                   ###########################
      #######################       GENERATIONS FOR LARVA       ###########################
      #######################                                   ###########################
      #####################################################################################
      #continue to increase the percentages
      if (lperc >= lcurr) { lcurr = lperc }
      # if it is the start of a new generation, reset the occurences to 0 (none of the % have been reached yet)
      else{
        lgen = lgen + 1 # increase the generation count
        ltf = 0
        lff = 0
        lsf = 0
        lcurr = 0
      }
      
      # if 25% reached
      if (lperc > .25 & ltf == 0)
      {ltf = 1 # check off that this occured
        col <- paste0(Lgen, lgen, "_", "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      # if 50 reached
      if (lperc > .50 & lff == 0){
        lff = 1 #check off that this occured
        col <- paste0(Lgen, lgen, "_", "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      # if 75 reached
      if (lperc > .75 & lsf == 0){
        lsf = 1 #check off that this occured
        col <- paste0(Lgen, lgen, "_", "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
    }
  }
  return(generations)
}

###############################################################################################################
###############################################################################################################
CodlingMothPercentPopulation <- function(CodMothParams, metdata_data.table, scale_shift=0) {
  # Number of stages which is 16: egg_1   thorugh egg_4,
  #                               Larva_1 thorugh Larva_4
  #                               Pupa_1  thorugh Pupa_4
  #                               Adult_1 thorugh Adult_4
  stage_gen_toiterate <- length(CodMothParams[, 1])
  
  # store relative numbers
  masterdata <- data.frame(metdata_data.table$dayofyear, 
                           metdata_data.table$year, 
                           metdata_data.table$month, 
                           metdata_data.table$Cum_dd_F)
  
  colnames(masterdata) <- c("dayofyear", "year", "month", "Cum_dd_F")
  
  for (i in 1:stage_gen_toiterate) {
    perc_num <- pweibull(metdata_data.table$Cum_dd_F, 
                         shape=CodMothParams[i,3], 
                         scale=CodMothParams[i,4] * (1 + scale_shift))
    perc_num <- data.frame(perc_num)
    colnames(perc_num) <- paste("Perc", CodMothParams[i, 1], "Gen", CodMothParams[i, 2], sep="")
    masterdata <- cbind(masterdata, perc_num)
  }
  rm(metdata_data.table)
  
  masterdata$PercEgg = 0
  masterdata$PercLarva = 0
  masterdata$PercPupa = 0
  masterdata$PercAdult = 0
  i = 1

  for (i in 1:4) {
    column_name <- paste("Perc", CodMothParams[i, 1], "Gen", CodMothParams[i,2], sep="")
    column_no <- which( colnames(masterdata)==column_name )
    # write.table(CodMothParams, paste0("/data/hydro/users/Hossein/", "test_param", "_1", ".txt"))
    masterdata$PercEgg[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                      masterdata$Cum_dd_F <= CodMothParams[i,6]] <- masterdata[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                                                                             masterdata$Cum_dd_F <= CodMothParams[i,6], 
                                                                             column_no]
  }
  for (i in 5:8) {
    column_name <- paste("Perc", CodMothParams[i,1], "Gen", CodMothParams[i,2], sep="")
    column_no <- which( colnames(masterdata)==column_name )
    # write.table(CodMothParams, paste0("/data/hydro/users/Hossein/", "test_param", "_5", ".txt"))
    masterdata$PercLarva[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                        masterdata$Cum_dd_F <= CodMothParams[i,6]] <- masterdata[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                                                                                 masterdata$Cum_dd_F <= CodMothParams[i,6], 
                                                                               column_no]
  }  
  for (i in 9:12) {
    column_name <- paste("Perc", CodMothParams[i,1], "Gen", CodMothParams[i,2], sep="")
    column_no <- which( colnames(masterdata)==column_name )
    masterdata$PercPupa[masterdata$Cum_dd_F>CodMothParams[i,5] & 
                       masterdata$Cum_dd_F<=CodMothParams[i,6]] <- masterdata[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                                                                            masterdata$Cum_dd_F <= CodMothParams[i,6], 
                                                                            column_no]
  } 
  for (i in 13:16) {
    column_name<-paste("Perc", CodMothParams[i,1], "Gen", CodMothParams[i,2], sep="")
    column_no<-which( colnames(masterdata)==column_name )
    masterdata$PercAdult[masterdata$Cum_dd_F > CodMothParams[i,5] & 
                        masterdata$Cum_dd_F <= CodMothParams[i,6]] <- masterdata[masterdata$Cum_dd_F > 
                        CodMothParams [i,5] & masterdata$Cum_dd_F <= CodMothParams[i,6], column_no]
  } 
  masterdata <- masterdata[, c("PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4",
                               "PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4",
                               "PercEgg", "PercLarva","PercPupa","PercAdult",
                               "dayofyear","year","month")]
  return(masterdata)
}
##############################################################################################################
##############################################################################################################
CodlingMothRelPopulation <- function(CodMothParams, metdata_data.table, scale_shift=0){
  # Number of stages which is 16: egg_1   thorugh egg_4,
  #                               Larva_1 thorugh Larva_4
  #                               Pupa_1  thorugh Pupa_4
  #                               Adult_1 thorugh Adult_4
  stage_gen_toiterate <- length(CodMothParams[, 1])

  # subset the data
  masterdata <- data.frame(metdata_data.table$dayofyear, metdata_data.table$year, 
                           metdata_data.table$month, metdata_data.table$Cum_dd_F)
  colnames(masterdata) <- c("dayofyear", "year", "month", "CumddF")

  # for combination of each stage and generation
  # produce relative population data
  for (i in 1:stage_gen_toiterate) {
    relnum <- dweibull(metdata_data.table$Cum_dd_F, 
                       shape=CodMothParams[i, 3], 
                       scale=CodMothParams[i, 4] * (1+scale_shift)) * 10000
    relnum <- data.frame(relnum)
    colnames(relnum) <- paste(CodMothParams[i, 1], "Gen", CodMothParams[i, 2], sep="")
    masterdata <- cbind(masterdata, relnum)
  }

  allrelnum <- masterdata
  rm(masterdata)

  # generate new columns, the total population of each stage 
  # across all generations
  allrelnum$SumEgg = allrelnum$EggGen1 + 
                     allrelnum$EggGen2 + 
                     allrelnum$EggGen3 + 
                     allrelnum$EggGen4
  
  allrelnum$SumLarva = allrelnum$LarvaGen1 + 
                       allrelnum$LarvaGen2 + 
                       allrelnum$LarvaGen3 + 
                       allrelnum$LarvaGen4

  allrelnum$SumPupa = allrelnum$PupaGen1 + 
                      allrelnum$PupaGen2 + 
                      allrelnum$PupaGen3 + 
                      allrelnum$PupaGen4

  allrelnum$SumAdult = allrelnum$AdultGen1 + 
                       allrelnum$AdultGen2 + 
                       allrelnum$AdultGen3 + 
                       allrelnum$AdultGen4

  allrelnum <- allrelnum[, c("LarvaGen1", "LarvaGen2", "LarvaGen3", "LarvaGen4", 
                             "AdultGen1", "AdultGen2", "AdultGen3", "AdultGen4", 
                             "SumEgg", "SumLarva", "SumPupa", "SumAdult", 
                             "dayofyear", "year", "month")]
  return(allrelnum)
}
###############################################################################################################
###############################################################################################################
# Function to compute and append gdd and cumulative gdd columns to metdata
add_dd_cumdd <- function(metdata_data.table, lower, upper) {
  twopi  = 2*pi
  pihalf = pi/2
  
  diff = metdata_data.table$tmax - metdata_data.table$tmin # column diff
  tsum = metdata_data.table$tmax + metdata_data.table$tmin # column tsum
  
  aveminlt= (tsum/2) - lower
  alpha1  = diff / 2
  avetemp = tsum / 2
  
  theta1 = asin((lower - avetemp) / alpha1)
  theta2 = asin((upper - avetemp) / alpha1)
  
  tmin = metdata_data.table$tmin
  tmax = metdata_data.table$tmax
  
  tempdata = data.frame(tmin, tmax, diff, tsum, aveminlt, alpha1, avetemp, theta1, theta2)
  tempdata$heat = 0

  # case 1 tmin >= upper threshold
  tempdata$heat[tempdata$tmin >= upper] = upper - lower
  
  # case 2 is handled by the default where heat is set to 0 initially it is when tmax<lower threshold
  
  # case 3 tmin>= lower threshold and the max <= upper threshold
  tempdata$heat[tempdata$tmin >= lower & tempdata$tmax <= upper] = tempdata$aveminlt[tempdata$tmin >= lower & tempdata$tmax <= upper]
  
  # case 4 tmin<lower threshold and tmax>lower and tmax<=upper threshold
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
###############################################################################################################
###############################################################################################################

#
#   Commented out on Feb. 19, 2021
#
# provide_time_stuff <- function(start_year, end_year, Nofvariables=4){
#   isLeapYear <- leap.year(seq(start_year, end_year))
#   countLeapYears <- length(isLeapYear[isLeapYear== TRUE])
#   nYears <- length(seq(start_year, end_year))
#   Nrecords <- 366 * countLeapYears + 365 * (nYears - countLeapYears ) #33603
#   Years <- seq(start_year, end_year)
#   ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
#   return(list(nYears, Nrecords, Nofvariables, Years, ind))
#   }
###############################################################################################################
###############################################################################################################
# function to create year month day dataframe to append to metdata dataframe
# create_ymdvalues <- function(nYears, Years, leap.year) {
#   daycount_in_year <- 0
#   moncount_in_year <- 0
#   yearrep_in_year <- 0

#   for(i in 1:nYears){
#      # chron::leap.year is a function however, here, Gridhar has abused the notation
#     ly<- leap.year(Years[i])
    
#     if(ly == TRUE){
#       days_in_mon <- c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
#     }
    
#     else{
#       days_in_mon <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
#     }
#     for( j in 1:12){
#       daycount_in_year <- c(daycount_in_year, seq(1, days_in_mon[j]))
#       moncount_in_year <- c(moncount_in_year, rep(j, days_in_mon[j]))
#       yearrep_in_year <- c(yearrep_in_year, rep(Years[i],days_in_mon[j]))
#     }
#   }
#   daycount_in_year <- daycount_in_year[-1] #delete the leading 0
#   moncount_in_year <- moncount_in_year[-1]
#   yearrep_in_year  <- yearrep_in_year[-1]
#   ymd <- cbind(yearrep_in_year, moncount_in_year, daycount_in_year)
#   colnames(ymd) <- c("year","month","day")
#   return(ymd)
# }
###############################################################################################################
###############################################################################################################
# readbinarydata_addmdy <- function(filename, Nrecords, Nofvariables=4, ymd, ind) {
#   fileCon = file(filename, "rb")
#   temp <- readBin(fileCon, integer(), size =2, n = Nrecords * Nofvariables, endian="little")
#   close(fileCon)

#   dataM <- matrix(0, Nrecords, Nofvariables)

#   dataM[1:Nrecords, 1] <- temp[ind]/40.00    # precip data
#   dataM[1:Nrecords, 2] <- temp[ind+1]/100.00 # Max temperature data
#   dataM[1:Nrecords, 3] <- temp[ind+2]/100.00 # Min temperature data
#   dataM[1:Nrecords, 4] <- temp[ind+3]/100.00 # Wind speed data
  
#   AllData <- cbind(ymd, dataM)
#   colnames(AllData) <- c("year", "month", "day", "precip", "tmax", "tmin", "winspeed")
#   return(AllData)
# }
###############################################################################################################
####################                                                                                     ######
####################  Phase 2: combining produced CM and CMPOP files and adding some more stuff to them. ######
####################                                                                                     ######
###############################################################################################################

merge_data <- function(input_dir, locations, models, file_prefix, emission){

  # emission should be one of observed, historical, rcp45, rcp85
  # models are either observed or one of the models.

  data = data.table()
  
  for(model in models){
    for(location in locations){
      if(model != "observed") {
        filename <- paste0(input_dir, "/", model, "/", emission, "/", file_prefix, "_", location)
      }
      else {
        filename <- paste0(input_dir, "/", file_prefix, "_" , location)
      }
    data_to_add = read.table(filename, header = TRUE, sep = ",")
    data <- rbind(data, data_to_add)
    }
  }
  return(data)
}

###############################################################################################################
###############################################################################################################
###############################################################################################################
##############                                                                                        
############## Phase 3: Using combined CM and CMPOP files to produce stuff, like vertdd, bloom, etc. 
##############                                                                                        
################################################################################################################
#####################################################################################
#######################                                            ##################
#######################                vertdd function             ##################
#######################                                            ##################
#####################################################################################
generate_vertdd <- function(input_dir, file_name, lower_temp = 4.5, upper_temp = 24.28){
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  lower = lower_temp
  upper = upper_temp
  twopi = 2 * pi
  pihlf = 0.5 * pi

  data$summ = data$tmin + data$tmax 
  data$diff = data$tmax - data$tmin
  data$diffsq = data$diff * data$diff

  data$b = 2 * upper - data$summ
  data$bsq = data$b * data$b

  data$a = 2 * lower - data$summ
  data$asq = data$a * data$a
  data$th1 = atan(data$a / sqrt(data$diffsq - data$asq))
  data$th2 = atan(data$b / sqrt(data$diffsq - data$bsq))

  data[tmin >= lower & tmax > upper, vertdd := ((-diff * cos(th2) - a * (th2 + pihlf))/twopi)]
  data[tmin >= lower & tmax <= upper, vertdd := summ/2 - lower]
  data[tmin < lower & tmax <= upper, vertdd := (diff * cos(th1) - (a * (pihlf - th1)))/twopi]
  data[tmin < lower & tmax > upper, vertdd := (-diff * (cos(th2) - cos(th1))-(a * (th2 - th1)))/twopi]
  data[tmin > tmax | tmax <= lower | tmin >= upper, vertdd := 0]

  data$summ = NULL
  data$diff = NULL
  data$diffsq = NULL
  data$b = NULL
  data$bsq = NULL
  data$a = NULL
  data$asq = NULL
  data$th1 = NULL
  data$th2 = NULL

  data = data[, vert_Cum_dd := cumsum(vertdd), by = list(location, model, time_period, year)] # latitude, longitude
  data$vert_Cum_dd_F = data$vert_Cum_dd * 1.8

  # The following three lines are from old parameters from Vince to Girid
  # data$cripps_pink = pnorm(data$vert_Cum_dd_F, mean = 495.51, sd = 42.58, lower.tail = TRUE)
  # data$gala = pnorm(data$vert_Cum_dd_F, mean = 528.56, sd = 41.95, lower.tail = TRUE)
  # data$red_deli = pnorm(data$vert_Cum_dd_F, mean = 522.74, sd = 42.79, lower.tail = TRUE)
  
  # the following is updated from new word document of Vince
  data$cripps_pink = pnorm(data$vert_Cum_dd_F, mean = 436.61, sd = 52.58, lower.tail = TRUE)
  data$gala = pnorm(data$vert_Cum_dd_F, mean = 468.99, sd = 49.49, lower.tail = TRUE)
  data$red_deli = pnorm(data$vert_Cum_dd_F, mean = 465.90, sd = 53.87, lower.tail = TRUE)

  return(data)
}
###################################################################
###################################################################
bloom_per_year_median_accross_models <- function(data_dt) {
  data_dt <- data_dt[, .(medDoY = as.integer(median(dayofyear))), 
                       by = c("time_period", "latitude", "longitude", 
                              "apple_type", "year")]
  return(data_dt)
}
bloom_per_year <- function(data, bloom_cut_off){
  # a note: if this is used in nay driver
  # the same as bloom function below. 
  # and we did take the medians out of bloom(.)
  # and created bloom_medians_across_models_time_periods(.)

}

bloom_medians_across_models_time_periods <- function(data){
  data <- data[, .(medDoY = as.integer(median(dayofyear))), 
               by = c("time_period", "latitude", "longitude", "apple_type")]
  return (data)
}

bloom <- function(data, bloom_cut_off){
  print (paste0("bloom_cut_off: ", bloom_cut_off))

  data <- subset(data, select = c("time_period",
                                  "latitude", "longitude", 
                                  "model", "year", "month", 
                                  "day", "dayofyear", 
                                  "cripps_pink", "gala", "red_deli"))
  
  data <- melt(data, id.vars = c("time_period",
                                 "longitude",  "latitude", 
                                 "model", "year", 
                                 "month", "day", "dayofyear"), 
              variable.name = "apple_type")
  
  # data = data[value >= 1.000000e+00,]
  print (paste0("line 1004 of bloom dim(data) = ", dim(data)))
  data = data[value >= bloom_cut_off, ]
  data = data[, head(.SD, 1), 
                by = c("time_period", "latitude", "longitude", 
                       "model", "year", "apple_type")]
  return (data)
}
#######################################################################################
#######################################################################################
# Generation Generator
##################################
generations_func <- function(input_dir, file_name){
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  generations_aug  <- data[data[, month==8 & day==23]]
  generations_nov <- data[data[,  month==11 & day==5]]

  generations_aug$NumAdultGens <- generations_aug$PercAdultGen1 + 
                                  generations_aug$PercAdultGen2 + 
                                  generations_aug$PercAdultGen3 + 
                                  generations_aug$PercAdultGen4
  
  generations_aug$NumLarvaGens <- generations_aug$PercLarvaGen1 + 
                                  generations_aug$PercLarvaGen2 + 
                                  generations_aug$PercLarvaGen3 + 
                                  generations_aug$PercLarvaGen4

  generations_nov$NumAdultGens <- generations_nov$PercAdultGen1 + 
                                  generations_nov$PercAdultGen2 + 
                                  generations_nov$PercAdultGen3 + 
                                  generations_nov$PercAdultGen4

  generations_nov$NumLarvaGens <- generations_nov$PercLarvaGen1 + 
                                  generations_nov$PercLarvaGen2 + 
                                  generations_nov$PercLarvaGen3 + 
                                  generations_nov$PercLarvaGen4
  return (list(generations_aug, generations_nov))
}

#####################################################################################
#######################                                   ###########################
#######################       Diapause Function           ###########################
#######################                                   ###########################
#####################################################################################
diapause_abs_rel <- function(data,
                             diap_param = c(102.6077, 1.306483, 16.95815),
                             shift = 0){
    ###
    ###    This is updated version of diapause_abs_rel_daylength().
    ###    It accepts the diapause parameters as input for "sensitivity" analysis.
    ###

    #
    #    Updated on Feb 24, 2021. I commented out 
    #    the file_name and data below and change the input to have
    #    the data directly. The drivers also should be updated.
    #
    #    The reason is that for sensitivity stuff I already combined all emissions
    #    and scale_shifts. So, I wanted to use one file for all of them.
    #
    # file_name = paste0(input_dir, file_name)
    # data <- data.table(readRDS(file_name))

    theta = 0.2163108 + (2 * atan(0.9671396 * tan(0.00860 * (data$dayofyear - 186))))
    phi = asin(0.39795 * cos(theta))
    D = 24 - ((24/pi) * acos((sin(6 * pi / 180) + 
              (sin(data$latitude * pi / 180) * sin(phi)))/(cos(data$latitude * pi / 180) * cos(phi))))
    data$daylength = D

    total_scaler = diap_param[1]
    exp_scaler = diap_param[2]
    model_shift = diap_param[3]

    data$diapause = total_scaler * exp(-exp(exp_scaler * (data$daylength - model_shift + shift )))
    data$diapause1 = data$diapause
    data[diapause1 > 100, diapause1 := 100]
    data$enterDiap = (data$diapause1/100) * data$SumLarva
    data$escapeDiap = data$SumLarva - data$enterDiap

    sub = data
    rm (data)
    startingpopulationfortheyear <- 1000
    # generation 1
    sub[, LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), 
          by=list(year, model, time_period, city) ] # latitude,longitude,
    sub$AbsPopLarvaGen1 <- sub$LarvaGen1RelFraction*startingpopulationfortheyear
    sub$AbsPopLarvaGen1Diap <- sub$AbsPopLarvaGen1*sub$diapause1/100
    sub$AbsPopLarvaGen1NonDiap <- sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

    # generation 2
    sub[, LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), 
         by = list(year, model, time_period, city)] # latitude, longitude
    sub[, AbsPopLarvaGen2 := LarvaGen2RelFraction * sum(AbsPopLarvaGen1NonDiap)*3.9, 
          by = list(year,model, time_period, city)] # latitude, longitude
    sub$AbsPopLarvaGen2Diap <- sub$AbsPopLarvaGen2 * sub$diapause1/100
    sub$AbsPopLarvaGen2NonDiap <- sub$AbsPopLarvaGen2 - sub$AbsPopLarvaGen2Diap

    # generation 3
    sub[, LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), 
          by =list(year,model, time_period, city)] # latitude, longitude,

    sub[, AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, 
          by =list(year,model, time_period, city) ] # latitude, longitude
    sub$AbsPopLarvaGen3Diap <- sub$AbsPopLarvaGen3*sub$diapause1/100
    sub$AbsPopLarvaGen3NonDiap <- sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

    # generation 4
    sub[, LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), 
          by =list(year, model, time_period, city)] # latitude, longitude,

    sub[, AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, 
          by =list(year, model, time_period, city)] # latitude, longitude,

    sub$AbsPopLarvaGen4Diap <- sub$AbsPopLarvaGen4*sub$diapause1/100
    sub$AbsPopLarvaGen4NonDiap <- sub$AbsPopLarvaGen4 - sub$AbsPopLarvaGen4Diap

    ### get totals similar to Sum Larva column, but abs numbers
    sub$AbsPopTotal <- sub$AbsPopLarvaGen1 + sub$AbsPopLarvaGen2 + sub$AbsPopLarvaGen3 + sub$AbsPopLarvaGen4
    sub$AbsPopDiap <- sub$AbsPopLarvaGen1Diap + sub$AbsPopLarvaGen2Diap + sub$AbsPopLarvaGen3Diap + sub$AbsPopLarvaGen4Diap
    sub$AbsPopNonDiap <- sub$AbsPopLarvaGen1NonDiap + sub$AbsPopLarvaGen2NonDiap + 
                         sub$AbsPopLarvaGen3NonDiap + sub$AbsPopLarvaGen4NonDiap

    sub1 = subset(sub, 
                  select = c(# "latitude", "longitude", 
                             "County", "city", 
                             "model", "time_period", 
                             "year", "dayofyear", "CumDDinF", 
                             "SumLarva", "enterDiap", 
                             "escapeDiap", "AbsPopTotal",
                             "AbsPopNonDiap","AbsPopDiap", "daylength"))
    # pre_diap_plot <- sub1
    # write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
    # saveRDS(sub1, paste0(write_dir, "pre_diap_plot_", emission, ".rds"))
    # daylength_vec = sub1$daylength # remain of the function diapause_abs_rel_daylength()
    sub1 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), RelNonDiap = mean(escapeDiap), 
                    AbsLarvaPop = mean(AbsPopTotal), AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), 
                    CumulativeDDF = mean(CumDDinF)), 
                    by = c("time_period", "city", "dayofyear")] # Kirti: add model and years here??!! if we do, then there is no mean!
    
    # sub1 = cbind(sub1, daylength_vec) # remain of the function diapause_abs_rel_daylength()
    RelData = subset(sub1, select = c("time_period", "city", 
                                      "dayofyear", "CumulativeDDF", 
                                      "RelLarvaPop", "RelDiap", "RelNonDiap"))

    RelData = melt(RelData, id = c("time_period", "city", "dayofyear", "CumulativeDDF")) # Kirti, why CumulativeDDF is here?
    
    #RelData[, Group := as.character()]
    RelData$Group = "0"
    temp1 = RelData[variable %in% c("RelLarvaPop", "RelDiap")]
    temp2 = RelData[variable %in% c("RelLarvaPop", "RelNonDiap")]
    temp1$Group = "Relative Larva Pop Vs Diapaused"
    temp2$Group = "Relative Larva Pop Vs NonDiapaused"
    RelData = rbind(temp1, temp2)
    ###
    ### Absolute Population
    ###
    AbsData = subset(sub1, select = c("time_period", "city", 
                                      "dayofyear", "CumulativeDDF", 
                                      "AbsLarvaPop", "AbsDiap", "AbsNonDiap")) # "daylength"
    
    AbsData = melt(AbsData, id = c("time_period", "city", "dayofyear", "CumulativeDDF"))
    #AbsData[, Group := as.character()]
    AbsData$Group = "0"
    temp1 = AbsData[variable %in% c("AbsLarvaPop", "AbsDiap")]
    temp2 = AbsData[variable %in% c("AbsLarvaPop", "AbsNonDiap")]
    temp1$Group = "Absolute Larva Pop Vs Diapaused"
    temp2$Group = "Absolute Larva Pop Vs NonDiapaused"
    AbsData = rbind(temp1, temp2)

    # return (list(RelData, AbsData, sub1, pre_diap_plot))
    return (list(RelData, AbsData, sub1))
}

################################################################################
#    Generate Map 1
################################################################################

generate_diapause_map1 <- function(input_dir, file_name, 
                                   param_dir, 
                                   CodMothParams_name, 
                                   location_group_name="LocationGroups.csv"){
  CodMothParams <- read.table(paste0(param_dir, CodMothParams_name), header=TRUE, sep=",")
  sub1 = diapause_map1_prep(input_dir, file_name, param_dir, location_group_name)
  
  group_vec = c("time_period", "city", "latitude", "longitude")

  sub2 = sub1[, .(RelPctDiap=(auc(CumulativeDDF, RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100, 
                  RelPctNonDiap = (auc(CumulativeDDF, RelNonDiap)/auc(CumulativeDDF, RelLarvaPop))*100, 
                  AbsPctDiap=(auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100, 
                  AbsPctNonDiap=(auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by=group_vec]
  
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctDiapGen1 = (auc(CumulativeDDF, RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctDiapGen2 = (auc(CumulativeDDF, RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctDiapGen3 = (auc(CumulativeDDF, RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctDiapGen4 = (auc(CumulativeDDF, RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctNonDiapGen1 = (auc(CumulativeDDF, RelNonDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctNonDiapGen2 = (auc(CumulativeDDF, RelNonDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctNonDiapGen3 = (auc(CumulativeDDF, RelNonDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctNonDiapGen4 = (auc(CumulativeDDF, RelNonDiap)/auc(CumulativeDDF, RelLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  #
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctDiapGen1 = (auc(CumulativeDDF, AbsDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctDiapGen2 = (auc(CumulativeDDF, AbsDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctDiapGen3 = (auc(CumulativeDDF, AbsDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctDiapGen4 = (auc(CumulativeDDF, AbsDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  #
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctNonDiapGen1 = (auc(CumulativeDDF, AbsNonDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctNonDiapGen2 = (auc(CumulativeDDF, AbsNonDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctNonDiapGen3 = (auc(CumulativeDDF, AbsNonDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctNonDiapGen4 = (auc(CumulativeDDF, AbsNonDiap)/auc(CumulativeDDF, AbsLarvaPop))*100), by = group_vec], by = group_vec, all.x = TRUE)
  return (sub2)
}
###############################################################################################################
###############################################################################################################
diapause_map1_prep <- function(input_dir, file_name, param_dir){
  file_name = paste0(input_dir, file_name, ".rds")
  data <- data.table(readRDS(file_name))

  theta = 0.2163108 + (2 * atan(0.9671396 * tan(0.00860 * (data$dayofyear - 186))))
  phi = asin(0.39795 * cos(theta))
  D = 24 - ((24/pi) * acos((sin(6 * pi / 180) + 
                      (sin(data$latitude * pi / 180) * sin(phi)))/(cos(data$latitude * pi / 180) * cos(phi))))
  data$daylength = D

  data$diapause = 102.6077 * exp(-exp(-(-1.306483) * (data$daylength - 16.95815)))
  data$diapause1 = data$diapause
  data[diapause1 > 100, diapause1 := 100]
  data$enterDiap = (data$diapause1/100) * data$SumLarva
  data$escapeDiap = data$SumLarva - data$enterDiap

  sub = data
  rm(data)
  startingpopulationfortheyear<-1000
  
  # generation1
  sub[, LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), 
        by =list(year, model, 
                 latitude, longitude, time_period) ]
  sub$AbsPopLarvaGen1 <- sub$LarvaGen1RelFraction*startingpopulationfortheyear
  sub$AbsPopLarvaGen1Diap <- sub$AbsPopLarvaGen1*sub$diapause1/100
  sub$AbsPopLarvaGen1NonDiap <- sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

  #generation2
  sub[,LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), 
       by = list(year, model, latitude, longitude, time_period, city)]
  sub[,AbsPopLarvaGen2 := LarvaGen2RelFraction * sum(AbsPopLarvaGen1NonDiap)*3.9, 
       by = list(year,model, latitude,longitude,time_period, city)]
  sub$AbsPopLarvaGen2Diap <- sub$AbsPopLarvaGen2 * sub$diapause1/100
  sub$AbsPopLarvaGen2NonDiap <- sub$AbsPopLarvaGen2 - sub$AbsPopLarvaGen2Diap

  #generation3
  sub[,LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), 
       by =list(year,model, latitude, longitude, time_period, city)]

  sub[,AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, 
       by =list(year,model, latitude,longitude,time_period, city) ]
  sub$AbsPopLarvaGen3Diap <- sub$AbsPopLarvaGen3*sub$diapause1/100
  sub$AbsPopLarvaGen3NonDiap <- sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

  #generation4
  sub[,LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), 
       by =list(year,model, latitude,longitude,time_period, city)]

  sub[,AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, 
       by =list(year,model, latitude,longitude,time_period, city)]

  sub$AbsPopLarvaGen4Diap <- sub$AbsPopLarvaGen4*sub$diapause1/100
  sub$AbsPopLarvaGen4NonDiap <- sub$AbsPopLarvaGen4- sub$AbsPopLarvaGen4Diap

  ### get totals similar to Sum Larva column, but abs numbers
  sub$AbsPopTotal <- sub$AbsPopLarvaGen1 + sub$AbsPopLarvaGen2 + sub$AbsPopLarvaGen3 + sub$AbsPopLarvaGen4
  sub$AbsPopDiap <- sub$AbsPopLarvaGen1Diap + sub$AbsPopLarvaGen2Diap + sub$AbsPopLarvaGen3Diap + sub$AbsPopLarvaGen4Diap
  sub$AbsPopNonDiap <- sub$AbsPopLarvaGen1NonDiap + sub$AbsPopLarvaGen2NonDiap + 
                      sub$AbsPopLarvaGen3NonDiap + sub$AbsPopLarvaGen4NonDiap
  
  print ("line 1275 of core")
  print (sort(colnames(sub)))
  # County was in the following, however, it seems
  # for all USA there is no county (at least I do not have it!)
  # so ... This is done on March 6th, for creating the features for analog
  # 
  sub = subset(sub, select = c("latitude", "longitude", 
                               "city", 
                               "model", "time_period", 
                               "year", "dayofyear", "CumDDinF", 
                               "SumLarva", "enterDiap", 
                               "escapeDiap", "AbsPopTotal",
                               "AbsPopNonDiap","AbsPopDiap"))

  sub = sub[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), 
                RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), 
                AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), 
                CumulativeDDF = mean(CumDDinF)), 
            by = c("time_period", "city", "latitude", "longitude", "dayofyear")]
  return (sub)
}
###############################################################################################################


######################
###################### Written for Sensitivity Analysis
######################
##################################################################
compute_cumdd_adult_emergence_mean <- function(input_dir, 
                                               file_name="combined_CMPOP_", 
                                               emission){
  ## This is for cumdd of adult emergence
  ##
  filename <- paste0(input_dir, file_name, emission, ".rds")
  data <- data.table(readRDS(filename))
  data <- subset(data, select = c("city", "latitude", "longitude", 
                                  "model", "time_period", 
                                  "year", "dayofyear", 
                                  "PercAdultGen1", "PercAdultGen2", 
                                  "PercAdultGen3", "PercAdultGen4"))
  data <- data[, .(AdultGen1 = mean(PercAdultGen1), AdultGen2 = mean(PercAdultGen2), 
                   AdultGen3 = mean(PercAdultGen3), AdultGen4 = mean(PercAdultGen4)), 
                by = c("city", 
                       "latitude", "longitude", 
                       "model", "time_period", 
                       "year", "dayofyear")]
  return (data)
}

compute_cumdd_adult_emergence_median <- function(input_dir, 
                                                 file_name="combined_CMPOP_", 
                                                 emission){
  filename <- paste0(input_dir, file_name, emission, ".rds")
  data <- data.table(readRDS(filename))
  data <- subset(data, select = c("city", "latitude", "longitude", 
                                  "model", "time_period", 
                                  "year", "dayofyear", 
                                  "PercAdultGen1", "PercAdultGen2", 
                                  "PercAdultGen3", "PercAdultGen4"))

  data <- data[, .(AdultGen1 = median(PercAdultGen1), AdultGen2 = median(PercAdultGen2), 
                   AdultGen3 = median(PercAdultGen3), AdultGen4 = median(PercAdultGen4)), 
                 by = c("city", 
                        "latitude", "longitude", 
                        "model", "time_period", 
                        "year", "dayofyear")]
  return (data)
}
###############################

generate_new_param_scale <- function(param_dir, param_name, scale_shift_percent){
  ## This function computes the low_gdd and high_gdd
  ## for the new modified scale of Weibull function.
  ## Keep in mind, qweibull is inverse of pweibull. 
  ## pinvweibull is NOT inverse of pweibull !!!
  params = read.table(paste0(param_dir, param_name), header=TRUE, sep=",")
  params_new = params
  params_new$scale = params$scale * (1 + scale_shift_percent)
  for (ii in seq(1:16)){
    x_gddhigh = params$gddhigh[ii]
    y_gddhigh = pweibull(x_gddhigh, shape=params$shape[ii], scale=params$scale[ii])
    params_new$gddhigh[ii] = round(qweibull(y_gddhigh, shape = params_new$shape[ii], scale= params_new$scale[ii]))

    x_gddlow = params$gddlow[ii]
    y_gddlow = pweibull(x_gddlow, shape=params$shape[ii], scale=params$scale[ii])
    params_new$gddlow[ii] = round(qweibull(y_gddlow, shape = params_new$shape[ii], scale= params_new$scale[ii]))
  }
  out_name = paste0(param_dir, "CodlingMothparameters_", scale_shift_percent, ".txt")
  write.table(params_new, out_name, sep=",", row.names = F, quote = FALSE)
}


###########################################################################
#########################
######################### LeafLet Maps
#########################
###########################################################################
#library(shiny)
#library(shinydashboard)
#library(shinyBS)
#library(rgdal)    # for readOGR and others
#library(maps)
#library(sp)       # for spatial objects
#library(leaflet)  # for interactive maps (NOT leafletR here)
#library(dplyr)    # for working with data frames
#library(ggplot2)  # for plotting
#library(data.table)
#library(reshape2)
#library(RColorBrewer)


constructMap <- function(mapLayerData, layerlist, palColumn, legendVals, title, gradient = "RdBu") {
  pal <- colorNumeric(
                      # palette = "Blues,Paired,Greens,Purples",
                      # check - https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/colorPaletteCheatsheet.pdf
                      palette = gradient, # Spectral_reverse, #"Spectral",
                      domain = legendVals
                      )
    myLabelFormat = function(..., dates=FALSE){ 
      if(dates){ function(type = "numeric", cuts){ 
                          format(as.Date(cuts, origin = "2018-01-01"), "%b %d")
                        }  
      } else {
        labelFormat(...)
      }
    } 

    map <- leaflet() %>% 
           addTiles() %>% 
           addProviderTiles(providers$CartoDB.Positron) %>% 
           setView(lat = 47.40, lng = -119.53, zoom = 7)
    
    for(i in 1:length(mapLayerData)){
                loc = tstrsplit(mapLayerData[[i]]$location, "_")
                mapLayerData[[i]]$latitude = as.numeric(unlist(loc[1]))
                mapLayerData[[i]]$longitude = as.numeric(unlist(loc[2]))
                map <- addCircleMarkers(map,
                                        data = mapLayerData[[i]],
                                        lat = mapLayerData[[i]]$latitude, lng = mapLayerData[[i]]$longitude,
                                        stroke = FALSE,
                                        radius = 7,
                                        fillOpacity = 0.9,
                                        color = ~pal(mapLayerData[[i]][, get(palColumn)]) #,
                                        #popup=paste0("<b>", title, ": </b>", 
                                        #              round(mapLayerData[[i]][, get(palColumn)], 1),
                                        #              "<br/><b>Latitude: </b> ", mapLayerData[[i]]$latitude, 
                                        #              "<br/><b>Longitude: </b> ", mapLayerData[[i]]$longitude)
                                        )
              }
    
    if(title == "Median Day of Year") { map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                                                        title = title,
                                                        labFormat = myLabelFormat(prefix = "  ", dates=TRUE),
                                                        opacity = 0.7) 
    } else { map = addLegend(map, "bottomleft", pal = pal, values = legendVals,
                             title = title,
                             labFormat = myLabelFormat(prefix = " "),
                             opacity = 0.7)
          }
    map
  }

# addTiles() %>% 
# addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png", attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>% 
# addTiles() %>% addProviderTiles(providers$Esri.NatGeoWorldMap, group = "Nat Geo") %>%
# addTiles() %>% addProviderTiles(providers$Stamen.TonerLite, group = "Toner Lite") %>%
# addProviderTiles(providers$CartoDB.Positron) %>% 


make_non_overlapping <- function(overlap_dt){
  old_time_periods <- c("2040s", "2060s", "2080s", "1979-2015")
  new_time_periods <- c("2026-2050", "2051-2075", "2076-2095", "1979-2015")

  ########## Historical
  hist <- overlap_dt %>% filter(time_period == "1979-2015")

  ########## 2040
  F1 <- overlap_dt %>% filter(time_period == "2040s")
  F1 <- F1 %>% filter(year>=2025 & year<=2050)
  F1$time_period <- new_time_periods[1]

  ########## 2060
  F2 <- overlap_dt %>% filter(time_period == "2060s")
  F2 <- F2 %>% filter(year>=2051 & year<=2075)
  F2$time_period <- new_time_periods[2]

  ########## 2080
  F3 <- overlap_dt %>% filter(time_period == "2080s")
  F3 <- F3 %>% filter(year>=2076)
  F3$time_period <- new_time_periods[3]

  non_overlap_dt <- rbind(hist, F1, F2, F3)
  non_overlap_dt$time_period <- factor(non_overlap_dt$time_period, levels=new_time_periods, order=T)

}
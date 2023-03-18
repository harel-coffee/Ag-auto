#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(ggplot2)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)


#
# this core worked to produce all the data to the producing 
# the plots point, where all plots but the boxplots were not produced
#


###############################
# FUNCTIONS
###############################

CodlingMothPercentPopulation<-function(CodMothParams,metdata_data.table) {
  stage_gen_toiterate<-length(CodMothParams[,1])
  # store relative numbers
  masterdata<-data.frame(metdata_data.table$dayofyear,metdata_data.table$year, metdata_data.table$month, metdata_data.table$Cum_dd_F)
  colnames(masterdata)<-c("dayofyear","year","month","Cum_dd_F")
  for (i in 1:stage_gen_toiterate) {
    relnum<-pweibull(metdata_data.table$Cum_dd_F, shape=CodMothParams[i,3], scale=CodMothParams[i,4]) 
    relnum<-data.frame(relnum)
    colnames(relnum)<-paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    masterdata<-cbind(masterdata,relnum)
    head(masterdata)
  }
  head(masterdata)
  #head(allrelnum)
  allrelnum<-masterdata
  allrelnum$PercEgg = 0
  allrelnum$PercLarva = 0
  allrelnum$PercPupa = 0
  allrelnum$PercAdult = 0
  i = 1
  # allrelnum[allrelnum$Cum_dd_F > CodMothParams [i,5] & allrelnum$Cum_dd_F <= 
  #                                 CodMothParams [i,6], 5] <- allrelnum[allrelnum$Cum_dd_F > 
  #                                 CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6], 
  #                                 paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")]
  # for (i in 1:4) {
  #   columnname<-paste("Perc", 
  #                      CodMothParams[i,1], 
  #                      "Gen",CodMothParams[i,2],sep="")
  #   columnnumber<-which( colnames(allrelnum)==columnname )
  #   # allrelnum$PercEgg[allrelnum$Cum_dd_F > CodMothParams [i,5] & 
  #                                               allrelnum$Cum_dd_F <= 
  #                                               CodMothParams [i,6]] <- allrelnum[allrelnum$Cum_dd_F > 
  #                                               CodMothParams [i,5] & allrelnum$Cum_dd_F <= 
  #                                               CodMothParams [i,6],columnnumber]
  #   allrelnum[allrelnum$Cum_dd_F > CodMothParams [i,5] & allrelnum$Cum_dd_F <= 
  #                                            CodMothParams [i,6], columnnumber] <- allrelnum[allrelnum$Cum_dd_F > 
  #                                            CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6],columnnumber]
  # }
  for (i in 1:4) {
    columnname <- paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    columnnumber <- which( colnames(allrelnum)==columnname )
    allrelnum$PercEgg[allrelnum$Cum_dd_F > CodMothParams [i,5] & 
                      allrelnum$Cum_dd_F <= CodMothParams [i,6]] <- allrelnum[allrelnum$Cum_dd_F > 
                                                                     CodMothParams [i,5] & allrelnum$Cum_dd_F <= 
                                                                     CodMothParams [i,6],columnnumber]
  }
  for (i in 5:8) {
    columnname <- paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    columnnumber <- which( colnames(allrelnum)==columnname )
    allrelnum$PercLarva[allrelnum$Cum_dd_F > CodMothParams [i,5] & 
                                                allrelnum$Cum_dd_F <= CodMothParams [i,6]] <- allrelnum[allrelnum$Cum_dd_F > 
                                                CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6],columnnumber]
    #check 0 or NA
    # allrelnum[!(allrelnum$Cum_dd_F > CodMothParams [i,5] & 
    # allrelnum$Cum_dd_F <= CodMothParams [i,6]), columnnumber] <- NA
  }  
  for (i in 9:12) {
    columnname<-paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    columnnumber<-which( colnames(allrelnum)==columnname )
    allrelnum$PercPupa[allrelnum$Cum_dd_F > CodMothParams [i,5] & allrelnum$Cum_dd_F <= 
                                         CodMothParams [i,6]] <- allrelnum[allrelnum$Cum_dd_F > 
                                        CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6],columnnumber]
  } 
  for (i in 13:16) {
    columnname<-paste("Perc",CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    columnnumber<-which( colnames(allrelnum)==columnname )
    allrelnum$PercAdult[allrelnum$Cum_dd_F > CodMothParams [i,5] & 
                             allrelnum$Cum_dd_F <= CodMothParams [i,6]] <- allrelnum[allrelnum$Cum_dd_F > 
                             CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6],columnnumber]
    #allrelnum[!(allrelnum$Cum_dd_F > CodMothParams [i,5] & allrelnum$Cum_dd_F <= CodMothParams [i,6]), columnnumber] <- NA
  } 
  head(allrelnum)
  allrelnum[,"PercAdult"]
  allrelnum<-allrelnum[, c( #"PercEggGen1", "PercEggGen2", "PercEggGen3", "PercEggGen4",
                            "PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4",
                            #"PercPupaGen1", "PercPupaGen2", "PercPupaGen3", "PercPupaGen4",
                            "PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4",
                            "PercEgg", "PercLarva","PercPupa","PercAdult",
                            "dayofyear","year","month")]
  return(allrelnum)
}

CodlingMothRelPopulation<-function(CodMothParams,metdata_data.table) {
  stage_gen_toiterate<-length(CodMothParams[,1])
  masterdata<-data.frame(metdata_data.table$dayofyear,metdata_data.table$year, 
                         metdata_data.table$month, metdata_data.table$Cum_dd_F)
  colnames(masterdata)<-c("dayofyear","year","month","CumddF")
  for (i in 1:stage_gen_toiterate) {
    relnum<-dweibull(metdata_data.table$Cum_dd_F, shape=CodMothParams[i,3], scale=CodMothParams[i,4]) * 10000
    relnum<-data.frame(relnum)
    colnames(relnum)<-paste(CodMothParams[i,1], "Gen",CodMothParams[i,2],sep="")
    masterdata<-cbind(masterdata,relnum)
    head(masterdata)
  }
  head(masterdata)
  allrelnum<-masterdata
  allrelnum$SumEgg = allrelnum$EggGen1 + allrelnum$EggGen2 + allrelnum$EggGen3 + allrelnum$EggGen4
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
  head(allrelnum)
  #allrelnum<-allrelnum[, c("SumEgg", "SumLarva","SumPupa","SumAdult","dayofyear","year","month")]
  allrelnum<-allrelnum[, c(#"EggGen1", "EggGen2", "EggGen3", "EggGen4", 
                           "LarvaGen1", "LarvaGen2", "LarvaGen3", "LarvaGen4", 
                           #"PupaGen1", "PupaGen2", "PupaGen3", "PupaGen4",
                           "AdultGen1", "AdultGen2", "AdultGen3", "AdultGen4", 
                           "SumEgg", "SumLarva","SumPupa","SumAdult","dayofyear","year","month")]
  return(allrelnum)
}

#  function to append gdd and cumulative gdd columns to met data
# add_dd_cumudd <- function(metdata_data.table, lower, upper) {
#   # temporary variables 
#   twice_pi = 2*pi
#   half_pi = pi/2
#   twice_lower = 2* lower   #fk1
#   gdd_temp1 = 0
#   twice_upper = 2* upper #fk1b
#   diffmaxmin = metdata_data.table$tmax - metdata_data.table$tmin  # column diff
#   summaxmin = metdata_data.table$tmax + metdata_data.table$tmin # column  tsum
#   twice_lower_minus_summaxmin = twice_lower - summaxmin   # d2
#   twice_upper_minus_summaxmin <- twice_upper - summaxmin # d2b
#   theta = atan(twice_lower_minus_summaxmin/sqrt(diffmaxmin^2 -twice_lower_minus_summaxmin^2))
#   theta2 = atan(twice_upper_minus_summaxmin/sqrt(diffmaxmin^2 -twice_upper_minus_summaxmin^2))
#   tmin<-metdata_data.table$tmin
#   tmax<-metdata_data.table$tmax
#   tempdata<-data.frame(tmin, tmax, diffmaxmin,summaxmin, twice_lower_minus_summaxmin, theta, twice_upper_minus_summaxmin, theta2)
#   tempdata[is.na(tempdata)]<--9999  
#   tempdata$theta[tempdata$theta> 0 & 
#                          tempdata$twice_lower_minus_summaxmin < 0] <- tempdata$theta[tempdata$theta> 0 & 
#                                                                       tempdata$twice_lower_minus_summaxmin < 0] - 
#                                                                       2 * half_pi
#   tempdata$theta2[tempdata$theta2> 0 & tempdata$twice_upper_minus_summaxmin < 0] <- tempdata$theta2[tempdata$theta2> 0 & 
#                                                                              tempdata$twice_upper_minus_summaxmin < 0] - 
#                                                                              2*half_pi
#   tempdata$heat<-0
#   tempdata$heat<-(tempdata$diffmaxmin*cos(tempdata$theta) - 
#                                       tempdata$twice_lower_minus_summaxmin * 
#                                                  (half_pi -tempdata$theta ))/ twice_pi
#   tempdata$heat[tempdata$tmin >= lower] <-(tempdata$summaxmin[tempdata$tmin >= lower] - twice_lower)/2
#   tempdata$heat2 <- tempdata$heat
#   tempdata$heat2<-(tempdata$diffmaxmin*cos(tempdata$theta2) - 
#                                         tempdata$twice_upper_minus_summaxmin*(half_pi - 
#                                         tempdata$theta2 ))/ twice_pi
#   tempdata$heat2 = tempdata$heat - tempdata$heat2
#   tempdata$dd <- -9999
#   tempdata$dd[tempdata$tmin > tempdata$tmax ] <- 0
#   tempdata$dd[tempdata$tmin >= upper ] <- upper-lower
#   tempdata$dd[tempdata$tmax <= lower ] <- 0
#   tempdata$dd[tempdata$tmin >= lower ] <- tempdata$heat[tempdata$tmin >= lower ]
#   tempdata$dd[tempdata$tmax > upper ] <- tempdata$heat2[tempdata$tmax > upper ]
#   tempdata$dd[tempdata$dd == -9999 ] <-tempdata$heat[tempdata$dd == -9999 ]
#   head(tempdata)
#   metdata_data.table$dd <-tempdata$dd
#   metdata_data.table[, Cum_dd := cumsum(dd), by=list(year)]
#   head(metdata_data.table)
#   return(metdata_data.table)
# }

#  function to append gdd and cumulative gdd columns to met data
add_dd_cumudd <- function(metdata_data.table, lower, upper) {
  twopi = 2 * pi
  pihalf = pi / 2
  diff = metdata_data.table$tmax - metdata_data.table$tmin # column diff
  tsum = metdata_data.table$tmax + metdata_data.table$tmin # column  tsum
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
  
  metdata_data.table$dd <-tempdata$heat
  metdata_data.table[, Cum_dd := cumsum(dd), by=list(year)]
  head(metdata_data.table)
  return(metdata_data.table)
}

# function to create year month day dataframe to append to metdata dataframe
create_ymdvalues <- function(nYears, Years, leap.year) {
  daycount_in_year<-0
  moncount_in_year<-0
  yearrep_in_year<-0
  for(i in 1:nYears){
    ly<-leap.year(Years[i])
    
    if(ly == TRUE){
      days_in_mon<-c(31,29,31,30,31,30,31,31,30,31,30,31)
    }
    
    else{
      days_in_mon<-c(31,28,31,30,31,30,31,31,30,31,30,31)
    }
    for( j in 1:12){
      daycount_in_year<-c(daycount_in_year,seq(1,days_in_mon[j]))
      moncount_in_year<-c(moncount_in_year,rep(j,days_in_mon[j]))
      yearrep_in_year<-c(yearrep_in_year,rep(Years[i],days_in_mon[j]))
    }
  }
  head(daycount_in_year)
  daycount_in_year<-daycount_in_year[-1] #delete the leading 0
  moncount_in_year<-moncount_in_year[-1]
  yearrep_in_year<-yearrep_in_year[-1]
  ymd<-cbind(yearrep_in_year,moncount_in_year,daycount_in_year)
  head(ymd)
  colnames(ymd)<-c("year","month","day")
  return(ymd)
}

# function to read binary data
readbinarydata_addmdy <- function(filename, Nrecords, Nofvariables, ymd, ind) {
  fileCon = file(filename, "rb")
  temp<-readBin(fileCon, integer(), size =2, n = Nrecords * Nofvariables, endian="little")
  dataM<-matrix(0, Nrecords, 4)
  k<-1
  dataM[1:Nrecords,1]<-temp[ind]/40.00  #precip data
  dataM[1:Nrecords,2]<-temp[ind+1]/100.00  #Max temperature data
  dataM[1:Nrecords,3]<-temp[ind+2]/100.00  #Min temperature data
  dataM[1:Nrecords,4]<-temp[ind+3]/100.00  #Wind speed data
  AllData<-cbind( ymd,dataM)
  head(AllData)
  # calculate daily GDD
  head(AllData)
  colnames(AllData)<-c("year","month","day","precip","tmax","tmin","winspeed")
  close(fileCon)
  return(AllData)
}

###################################
## Script
###################################

calcPopulation <- function(filename, input_folder, output_folder)
{
  outfile <- paste0(output_folder, paste0("CM", strsplit(filename, "data")[[1]][2]))
  print(outfile)  
  #start time in the met data
  data_start_year<-1979  
  data_start_month<-1
  data_start_day<-1
  data_end_year<-2015 #end time in the data
  
  # get number of records and number years, indices of variables 
  ## indices just denote the fact that, given n variables, a new day's data starts every nth read 
  isLeapYear <-leap.year(seq(data_start_year,data_end_year))
  countLeapYears <- length( isLeapYear[isLeapYear== TRUE])
  nYears<-length(seq(data_start_year,data_end_year))
  Nrecords<-366 * countLeapYears +365 * (nYears - countLeapYears ) #33603
  Nofvariables <- 4 #number of varaibles or column in the forcing data file
  Years<-seq(data_start_year,data_end_year)
  ind<-seq(1,Nrecords*Nofvariables, Nofvariables)
  
  ## create year, month, day values based on start year, number of years and leap year info
  ymd<-create_ymdvalues (nYears, Years, leap.year)
  
  ## read met data and add year month day variables
  inputfilenameandpath<-paste(input_folder,filename,sep="")
  metdata<-readbinarydata_addmdy(inputfilenameandpath,Nrecords, Nofvariables, ymd, ind)
  head(metdata)
  metdata_data.table <- data.table(metdata)
  ## set upper and lower temperature bounds
  lower <-10 # 50 F
  upper<-31.11  # 88F
  ## calculate daily and cumulative gdd to met data
  metdata_data.table<-add_dd_cumudd(metdata_data.table, lower, upper)
  head(metdata_data.table)
  # convert celcius to farenheit
  metdata_data.table$Cum_dd_F = metdata_data.table$Cum_dd *1.8
  head(metdata_data.table$Cum_dd_F)
  
  # add day of year from 1 to 365/366 depending on year
  metdata_data.table$dum<-1 # dummy
  head(metdata_data.table)
  metdata_data.table[, dayofyear := cumsum(dum), by=list(year)]
  head(metdata_data.table)
  
  ### GET RELATIVE POPULATION DISTRIBUTIONS
  CodMothParams<-read.table("CodlingMothparameters.txt", header=TRUE, sep=",")
  relpopulation<- CodlingMothRelPopulation(CodMothParams,metdata_data.table)
  toprint<-cbind(metdata_data.table$tmax, 
                metdata_data.table$tmin, 
                metdata_data.table$dd, 
                metdata_data.table$Cum_dd, 
                metdata_data.table$Cum_dd_F, 
                relpopulation, 
                metdata_data.table$day )
  colnames(toprint)<-c("tmax", "tmin", "DailyDD", 
                       "CumDDinC","CumDDinF", colnames(relpopulation)[1:8],
                        "SumEgg", "SumLarva", "SumPupa", 
                        "SumAdult", "dayofyear", 
                        "year", "month", "day")
  head(toprint)
  
  ## GET PERCENT population distributions
  percpopulation<- CodlingMothPercentPopulation(CodMothParams,metdata_data.table)
  head(percpopulation)
  #toprint<-cbind(percpopulation$PercEgg, percpopulation$PercLarva,percpopulation$PercPupa,percpopulation$PercAdult,toprint)
  toprint <- cbind(percpopulation[,1:12], toprint)
  # colnames(toprint)<-c("PercEgg","PercLarva","PercPupa", 
  #                      "PercAdult", "tmax", "tmin", "DailyDD", "CumDDinC",
  #                      "CumDDinF","SumEgg","SumLarva","SumPupa","SumAdult","dayofyear","year","month","day")
  colnames(toprint)<-c(colnames(percpopulation)[1:8], 
                       "PercEgg", "PercLarva", "PercPupa", 
                       "PercAdult", 
                       "tmax", "tmin", "DailyDD", 
                       "CumDDinC", "CumDDinF", 
                       colnames(relpopulation)[1:8], 
                       "SumEgg", "SumLarva", "SumPupa", 
                       "SumAdult", "dayofyear", 
                       "year", "month", "day")
  #toprint_reorder<-toprint[,c(15:17,14,5:9,10:13,1:4)]
  toprint_reorder<-toprint[,c(31:33,30,13:17,18:29,1:12)]
  head(toprint_reorder)
  
  ####### THE Following is not used anywhere. Moreover, it seems the second and third line 
  ####### is a bug! should not they use setofrownumbersAdult50 and setofrownumbersAdult75?
  # setofrownumbersLarva25<-1
  # setofrownumbersLarva50<-1
  # setofrownumbersLarva75<-1
  # setofrownumbersAdult25<-1
  # setofrownumbersAdult50<-1
  # setofrownumbersAdult75<-1

  # numberofrecords=length(data[,1])
  # for ( i in 2:numberofrecords ) {
    
  #  if (data[i, "PercAdult"] >= .25 & data[i-1, "PercAdult"] < .25 ) {
  #    setofrownumbersAdult25<-c(setofrownumbersAdult25, i)
  #  }
  #  if (data[i, "PercAdult"] >= .5 & data[i-1, "PercAdult"] < .50 ) {
  #    setofrownumbersAdult50<-c(setofrownumbersAdult25, i)
  #  }
  #  if (data[i, "PercAdult"] >= .75 & data[i-1, "PercAdult"] < .75 ) {
  #    setofrownumbersAdult75<-c(setofrownumbersAdult25, i)
  #  }
  #}

  #setofrownumbersAdult25
  #toprint[setofrownumbersAdult25, c(1:12,30:33)]
  #toprint[33453:33457, c(1:12,30:33)]
  ## print data
  
  #for each year
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
  
  for (i in min(toprint_reorder$year):max(toprint_reorder$year))
  {
    Agen = "AGen"
    Lgen = "LGen"
    
    atf = 0 #25 occured adult
    aff = 0 #50 occured adult
    asf = 0 #75 occured adult
    acurr = 0 #current % population per generation Adult
    ltf = 0 #25 occured larva
    lff = 0 #50 occured larva
    lsf = 0 #75 occured larva
    lcurr = 0 #current % population per generation Larva
    agen = 1 #generation of the year adults
    lgen = 1 #generation of the year for larva
    
    em = 0 #emergence occured
    
    #add each year to dataframe
    generations[nrow(generations) + 1, 1] <- i
    
    #for each day of the year
    for (j in 1:nrow(subset(toprint_reorder, year == i)))
    {
      #get the data for each day
      row <- subset(toprint_reorder, year == i)[j, ]
      aperc <- row$PercAdult #current % adults
      aGen1perc <- row$PercAdultGen1
      aGen2perc <- row$PercAdultGen2
      aGen3perc <- row$PercAdultGen3
      aGen4perc <- row$PercAdultGen4
      lperc <- row$PercLarva #current % larva
      lGen1perc <- row$PercLarvaGen1
      lGen2perc <- row$PercLarvaGen2
      lGen3perc <- row$PercLarvaGen3
      lGen4perc <- row$PercLarvaGen4
      
      ########## DiaPause ########## 
      
      if (row$month == 8 & row$day == 13)
      {
        generations[generations$year == i,]["Diapause"] <- lperc
      }
      #######Percentages at the start of each month ############
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
      if (row$month == 9 & row$day == 15)
        
      {     
        
        generations[generations$year == i,]["LarvaSep15"] <- lperc
        
      }
      
      if (row$month == 10 & row$day == 1)
        
      {
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
      
      if (row$month == 11 & row$day == 1)
        
      {
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
      
      if (row$month == 12 & row$day == 1)
        
      {
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
      
      
      ########## Emergence ########## 
      
      if (row$CumDDinC > 100 & em == 0)
      {
        em = 1
        generations[generations$year == i,]["Emergence"] <- row$dayofyear
      }
      
      ########## generations for adults ############
      
      #continue to increase the percentage
      if (aperc >= acurr) { acurr = aperc }
      
      #if there is a reset (start of a new generation)
      else
      {
        agen = agen + 1 #increase generation count
        
        #now reset boolean for occuring percentages (for new generation)
        atf = 0
        aff = 0
        asf = 0
        acurr = 0
      }
      
      #if new gen reached 25
      if (aperc > .25 & atf == 0)
      {
        atf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if new gen reached 50
      if (aperc > .50 & aff == 0)
      {
        aff = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if new gen reached 75
      if (aperc > .75 & asf == 0)
      {
        asf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      ########## generations for larva ############
      
      #continue to increase the percentages
      if (lperc >= lcurr) { lcurr = lperc }
      
      #if it is the start of a new generation, reset the occurences to 0 (none of the % have been reached yet)
      else
      {
        lgen = lgen + 1 #increase the generation count
        ltf = 0
        lff = 0
        lsf = 0
        lcurr = 0
      }
      
      #if 25% reached
      if (lperc > .25 & ltf == 0)
      {
        ltf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if 50 reached
      if (lperc > .50 & lff == 0)
      {
        lff = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if 75 reached
      if (lperc > .75 & lsf == 0)
      {
        lsf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
    }
  }
  #tempfilename<-paste(outfile,"_alldata",sep="");
  #write.table(toprint,file=tempfilename,sep=",",quote=FALSE,col.names=TRUE,row.names=FALSE)  
  write.table(generations, file=outfile, sep=",",quote=FALSE,col.names=TRUE,row.names=FALSE)
}

mothPopulation <- function(input_folder, 
                           list_file = "list.txt", 
                           output_folder)
{
  #setwd(input_folder)
  
  list_file = file(list_file, open="r")
  files = readLines(list_file) 
  num_files = length(files)
  
  for (i in 1:num_files) 
  {
    #inputfilenameandpath<- paste(input_folder,files[i], sep="")
    calcPopulation(files[i],input_folder, output_folder) 
  }
  
  close(list_file)
}


getPercPopulation <- function(filename, input_folder, start_year, end_year)
{
  #start time in the met data
  data_start_year<-start_year  
  data_start_month<-1
  data_start_day<-1
  data_end_year<-end_year #end time in the data
  
  # get number of records and number years, indices of variables 
  ## indices just denote the fact that, given n variables, a new day's data starts every nth read 
  isLeapYear <-leap.year(seq(data_start_year,data_end_year))
  countLeapYears <- length( isLeapYear[isLeapYear== TRUE])
  nYears<-length(seq(data_start_year,data_end_year))
  Nrecords<-366 * countLeapYears +365 * (nYears - countLeapYears ) #33603
  Nofvariables <- 4 #number of varaibles or column in the forcing data file
  Years<-seq(data_start_year,data_end_year)
  ind<-seq(1,Nrecords*Nofvariables, Nofvariables)
  
  ## create year, month, day values based on start year, number of years and leap year info
  ymd<-create_ymdvalues (nYears, Years, leap.year)
  
  ## read met data and add year month day variables
  inputfilenameandpath<-paste(input_folder,filename,sep="")
  metdata<-readbinarydata_addmdy(inputfilenameandpath,Nrecords, Nofvariables, ymd, ind)
  head(metdata)
  metdata_data.table <- data.table(metdata)
  ## set upper and lower temperature bounds
  lower <-10 # 50 F
  upper<-31.11  # 88F
  ## calculate daily and cumulative gdd to met data
  metdata_data.table<-add_dd_cumudd(metdata_data.table, lower, upper)
  head(metdata_data.table)
  # convert celcius to farenheit
  metdata_data.table$Cum_dd_F = metdata_data.table$Cum_dd *1.8
  head(metdata_data.table$Cum_dd_F)
  
  # add day of year from 1 to 365/366 depending on year
  metdata_data.table$dum<-1 # dummy
  head(metdata_data.table)
  metdata_data.table[, dayofyear := cumsum(dum), by=list(year)]
  head(metdata_data.table)
  
  CodMothParams<-read.table(paste0(input_folder, "CodlingMothparameters.txt"),header=TRUE,sep=",")
  ## GET PERCENT population distributions
  percpopulation <- CodlingMothPercentPopulation(CodMothParams,metdata_data.table)
  return(percpopulation)
}

getRelPopulation <- function(filename, input_folder, start_year, end_year)
{
  #start time in the met data
  data_start_year<-start_year  
  data_start_month<-1
  data_start_day<-1
  data_end_year<-end_year #end time in the data
  
  # get number of records and number years, indices of variables 
  ## indices just denote the fact that, given n variables, a new day's data starts every nth read 
  isLeapYear <-leap.year(seq(data_start_year,data_end_year))
  countLeapYears <- length( isLeapYear[isLeapYear== TRUE])
  nYears<-length(seq(data_start_year,data_end_year))
  Nrecords<-366 * countLeapYears +365 * (nYears - countLeapYears ) #33603
  Nofvariables <- 4 #number of varaibles or column in the forcing data file
  Years<-seq(data_start_year,data_end_year)
  ind<-seq(1,Nrecords*Nofvariables, Nofvariables)
  
  ## create year, month, day values based on start year, number of years and leap year info
  ymd<-create_ymdvalues (nYears, Years, leap.year)
  
  ## read met data and add year month day variables
  inputfilenameandpath<-paste(input_folder,filename,sep="")
  metdata<-readbinarydata_addmdy(inputfilenameandpath,Nrecords, Nofvariables, ymd, ind)
  head(metdata)
  metdata_data.table <- data.table(metdata)
  ## set upper and lower temperature bounds
  lower <-10 # 50 F
  upper<-31.11  # 88F
  ## calculate daily and cumulative gdd to met data
  metdata_data.table<-add_dd_cumudd(metdata_data.table, lower, upper)
  head(metdata_data.table)
  # convert celcius to farenheit
  metdata_data.table$Cum_dd_F = metdata_data.table$Cum_dd *1.8
  head(metdata_data.table$Cum_dd_F)
  
  # add day of year from 1 to 365/366 depending on year
  metdata_data.table$dum<-1 # dummy
  head(metdata_data.table)
  metdata_data.table[, dayofyear := cumsum(dum), by=list(year)]
  head(metdata_data.table)
  
  CodMothParams<-read.table(paste0(input_folder, "CodlingMothparameters.txt"),header=TRUE,sep=",")
  ## GET PERCENT population distributions
  relpopulation <- CodlingMothRelPopulation(CodMothParams,metdata_data.table)
  return(relpopulation)
}

prepareData_CMPOP <- function(filename, input_folder, start_year, end_year) {
  #start time in the met data
  data_start_year<-start_year  
  data_start_month<-1
  data_start_day<-1
  data_end_year<-end_year #end time in the data
  
  # get number of records and number years, indices of variables 
  # indices just denote the fact that, given n variables, a new day's data starts every nth read 
  isLeapYear <-leap.year(seq(data_start_year,data_end_year))
  countLeapYears <- length( isLeapYear[isLeapYear== TRUE])
  nYears <- length(seq(data_start_year,data_end_year))
  Nrecords <- 366 * countLeapYears +365 * (nYears - countLeapYears ) #33603
  Nofvariables <- 4 #number of varaibles or column in the forcing data file
  Years <- seq(data_start_year, data_end_year)
  ind <- seq(1,Nrecords*Nofvariables, Nofvariables)
  
  # create year, month, day values based on start year, number of years and leap year info
  ymd <- create_ymdvalues (nYears, Years, leap.year)
  
  ## read met data and add year month day variables
  inputfilenameandpath<-paste(input_folder,filename,sep="")
  metdata <- readbinarydata_addmdy(inputfilenameandpath,Nrecords, Nofvariables, ymd, ind)
  head(metdata)
  metdata_data.table <- data.table(metdata)
  ## set upper and lower temperature bounds
  lower <- 10 # 50 F
  upper <- 31.11  # 88F
  ## calculate daily and cumulative gdd to met data
  metdata_data.table <- add_dd_cumudd(metdata_data.table, lower, upper)
  head(metdata_data.table)
  # convert celcius to farenheit
  metdata_data.table$Cum_dd_F = metdata_data.table$Cum_dd *1.8
  head(metdata_data.table$Cum_dd_F)
  
  # add day of year from 1 to 365/366 depending on year
  metdata_data.table$dum<-1 # dummy
  head(metdata_data.table)
  metdata_data.table[, dayofyear := cumsum(dum), by=list(year)]
  head(metdata_data.table)
  
  CodMothParams<-read.table(paste0(input_folder, "CodlingMothparameters.txt"),header=TRUE,sep=",")
  
  #Relative Population
  relpopulation <- CodlingMothRelPopulation(CodMothParams,metdata_data.table)
  
  data <- cbind(metdata_data.table$tmax,metdata_data.table$tmin, 
                metdata_data.table$dd, metdata_data.table$Cum_dd, 
                metdata_data.table$Cum_dd_F, relpopulation, 
                metdata_data.table$day )
  colnames(data)<-c("tmax", "tmin", "DailyDD", "CumDDinC", "CumDDinF", 
                    colnames(relpopulation)[1:8], 
                    "SumEgg", "SumLarva", "SumPupa",
                    "SumAdult", "dayofyear", 
                    "year", "month", "day")
    
  #Percent Population
  percpopulation <- CodlingMothPercentPopulation(CodMothParams,metdata_data.table)
  
  data <- cbind(percpopulation[,1:12], data)
  colnames(data) <- c(colnames(percpopulation)[1:8], "PercEgg", 
                      "PercLarva", "PercPupa", "PercAdult", 
                      "tmax", "tmin", "DailyDD", "CumDDinC", 
                      "CumDDinF", colnames(relpopulation)[1:8], 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  #toprint_reorder<-toprint[,c(15:17,14,5:9,10:13,1:4)]
  data <- data[,c(31:33,30,13:17,18:29,1:12)]
  head(data)
  return(data)
}


prepareData_CM <- function(filename, input_folder, start_year, end_year) {
  #start time in the met data
  data_start_year <- start_year  
  data_start_month <- 1
  data_start_day <- 1
  data_end_year <- end_year #end time in the data
  
  # get number of records and number years, indices of variables 
  ## indices just denote the fact that, given n variables, a new day's data starts every nth read 
  isLeapYear <-leap.year(seq(data_start_year,data_end_year))
  countLeapYears <- length(isLeapYear[isLeapYear== TRUE])
  nYears <- length(seq(data_start_year,data_end_year))
  Nrecords <- 366 * countLeapYears + 365 * (nYears - countLeapYears ) #33603
  Nofvariables <- 4 #number of varaibles or column in the forcing data file
  Years <- seq(data_start_year, data_end_year)
  ind <- seq(1, Nrecords * Nofvariables, Nofvariables)
  
  ## create year, month, day values based on start year, number of years and leap year info
  ymd <- create_ymdvalues (nYears, Years, leap.year)
  
  ## read met data and add year month day variables
  inputfilenameandpath <- paste(input_folder,filename,sep="")
  metdata <- readbinarydata_addmdy(inputfilenameandpath,Nrecords, Nofvariables, ymd, ind)
  head(metdata)
  metdata_data.table <- data.table(metdata)
  ## set upper and lower temperature bounds
  lower <- 10 # 50 F
  upper <- 31.11  # 88F
  ## calculate daily and cumulative gdd to met data
  metdata_data.table <- add_dd_cumudd(metdata_data.table, lower, upper)
  head(metdata_data.table)
  # convert celcius to farenheit
  metdata_data.table$Cum_dd_F = metdata_data.table$Cum_dd *1.8
  head(metdata_data.table$Cum_dd_F)
  
  # add day of year from 1 to 365/366 depending on year
  metdata_data.table$dum <- 1 # dummy
  head(metdata_data.table)
  metdata_data.table[, dayofyear := cumsum(dum), by=list(year)]
  head(metdata_data.table)
  
  CodMothParams <- read.table(paste0(input_folder, "CodlingMothparameters.txt"),header=TRUE,sep=",")
  
  #Relative Population
  relpopulation <- CodlingMothRelPopulation(CodMothParams,metdata_data.table)
  
  data <- cbind(metdata_data.table$tmax,metdata_data.table$tmin, metdata_data.table$dd, 
                metdata_data.table$Cum_dd, metdata_data.table$Cum_dd_F, 
                relpopulation, metdata_data.table$day )
  colnames(data) <- c("tmax", "tmin", 
                      "DailyDD", "CumDDinC", "CumDDinF", 
                      colnames(relpopulation)[1:8], 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
    
  #Percent Population
  percpopulation <- CodlingMothPercentPopulation(CodMothParams,metdata_data.table)
  
  data <- cbind(percpopulation[,1:12], data)
  colnames(data) <- c(colnames(percpopulation)[1:8], "PercEgg", 
                      "PercLarva", "PercPupa", "PercAdult", 
                      "tmax", "tmin", "DailyDD", "CumDDinC", 
                      "CumDDinF", colnames(relpopulation)[1:8], 
                      "SumEgg", "SumLarva", "SumPupa", 
                      "SumAdult", "dayofyear", 
                      "year", "month", "day")
  #toprint_reorder<-toprint[,c(15:17,14,5:9,10:13,1:4)]
  data <- data[,c(31:33,30,13:17,18:29,1:12)]
  head(data)
  #return(data)
  setofrownumbersLarva25<-1
  setofrownumbersLarva50<-1
  setofrownumbersLarva75<-1
  setofrownumbersAdult25<-1
  setofrownumbersAdult50<-1
  setofrownumbersAdult75<-1
  numberofrecords=length(data[,1])
  for ( i in 2:numberofrecords ) {
    
    if (data[i, "PercAdult"] >= .25 & data[i-1, "PercAdult"] < .25 ) {
      setofrownumbersAdult25<-c(setofrownumbersAdult25, i)
    }
    if (data[i, "PercAdult"] >= .5 & data[i-1, "PercAdult"] < .50 ) {
      setofrownumbersAdult50<-c(setofrownumbersAdult25, i)
    }
    if (data[i, "PercAdult"] >= .75 & data[i-1, "PercAdult"] < .75 ) {
      setofrownumbersAdult75<-c(setofrownumbersAdult25, i)
    }
  }
  #setofrownumbersAdult25
  #data[setofrownumbersAdult25, c(1:12,30:33)]
  #data[33453:33457, c(1:12,30:33)]
  ## print data
  
  #for each year
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
    
    atf = 0 #25 occured adult
    aff = 0 #50 occured adult
    asf = 0 #75 occured adult
    acurr = 0 #current % population per generation Adult
    ltf = 0 #25 occured larva
    lff = 0 #50 occured larva
    lsf = 0 #75 occured larva
    lcurr = 0 #current % population per generation Larva
    agen = 1 #generation of the year adults
    lgen = 1 #generation of the year for larva
    
    em = 0 #emergence occured
    
    #add each year to dataframe
    generations[nrow(generations) + 1, 1] <- i
    
    #for each day of the year
    for (j in 1:nrow(subset(data, year == i)))
    {
      #get the data for each day
      row <- subset(data, year == i)[j, ]
      aperc <- row$PercAdult #current % adults
      aGen1perc <- row$PercAdultGen1
      aGen2perc <- row$PercAdultGen2
      aGen3perc <- row$PercAdultGen3
      aGen4perc <- row$PercAdultGen4
      lperc <- row$PercLarva #current % larva
      lGen1perc <- row$PercLarvaGen1
      lGen2perc <- row$PercLarvaGen2
      lGen3perc <- row$PercLarvaGen3
      lGen4perc <- row$PercLarvaGen4
      
      ########## DiaPause ########## 
      
      if (row$month == 8 & row$day == 13)
      {
        generations[generations$year == i,]["Diapause"] <- lperc
      }
      #######Percentages at the start of each month ############
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
      if (row$month == 9 & row$day == 15)
        
      {     
        
        generations[generations$year == i,]["LarvaSep15"] <- lperc
        
      }
      
      if (row$month == 10 & row$day == 1)
        
      {
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
      
      if (row$month == 11 & row$day == 1)
        
      {
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
      
      if (row$month == 12 & row$day == 1)
        
      {
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
      
      
      ########## Emergence ########## 
      
      if (row$CumDDinC > 100 & em == 0)
      {
        em = 1
        generations[generations$year == i,]["Emergence"] <- row$dayofyear
      }
      
      ########## generations for adults ############
      
      #continue to increase the percentage
      if (aperc >= acurr) { acurr = aperc }
      
      #if there is a reset (start of a new generation)
      else
      {
        agen = agen + 1 #increase generation count
        
        #now reset boolean for occuring percentages (for new generation)
        atf = 0
        aff = 0
        asf = 0
        acurr = 0
      }
      
      #if new gen reached 25
      if (aperc > .25 & atf == 0)
      {
        atf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if new gen reached 50
      if (aperc > .50 & aff == 0)
      {
        aff = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if new gen reached 75
      if (aperc > .75 & asf == 0)
      {
        asf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Agen, agen), "_"), "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      ########## generations for larva ############
      
      #continue to increase the percentages
      if (lperc >= lcurr) { lcurr = lperc }
      
      #if it is the start of a new generation, reset the occurences to 0 (none of the % have been reached yet)
      else
      {
        lgen = lgen + 1 #increase the generation count
        ltf = 0
        lff = 0
        lsf = 0
        lcurr = 0
      }
      
      #if 25% reached
      if (lperc > .25 & ltf == 0)
      {
        ltf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.25")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if 50 reached
      if (lperc > .50 & lff == 0)
      {
        lff = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.5")
        generations[generations$year == i,][col] <- row$dayofyear
      }
      
      #if 75 reached
      if (lperc > .75 & lsf == 0)
      {
        lsf = 1 #check off that this occured
        col <- paste0(paste0(paste0(Lgen, lgen), "_"), "0.75")
        generations[generations$year == i,][col] <- row$dayofyear
      }
    }
  }
  #tempfilename<-paste(outfile,"_alldata",sep="");
  #write.table(data,file=tempfilename,sep=",",quote=FALSE,col.names=TRUE,row.names=FALSE)  
  #write.table(generations, file=outfile, sep=",",quote=FALSE,col.names=TRUE,row.names=FALSE)
  return(generations)
}


#####
merge_add_countyGroup <- function(input_dir, param_dir, 
                                  locations_file_name,
                                  locationGroup_fileName,
                                  categories, file_prefix, version){

  merged_data <- merge_data(input_dir, param_dir, categories, locations_file_name, file_prefix, version)
  merged_data = add_countyGroup(merged_data, param_dir, loc_group_file_name= locationGroup_fileName)
  return(merged_data)
}

merge_data <- function(input_dir, param_dir, categories, locations_file_name, file_prefix, version){
  data = data.table()
  conn = file(paste0(param_dir, locations_file_name), open = "r")
  locations = readLines(conn)
  close(conn)

  for( category in categories) {
      for( location in locations) {
        if(category != "historical") {
          filename <- paste0(read_data_dir, "future_", file_prefix, "/", category, "/", version, "/", file_prefix, "_", location)
        }
        else {
          filename <- paste0(read_data_dir, "historical_", file_prefix, "/", file_prefix, "_" ,location)
            }
    data <- rbind(data, read.table(filename, header = TRUE, sep = ","))
      }
  }
  return(data)
}

add_countyGroup <- function(data, param_dir, loc_group_file_name){
  loc_grp = data.table(read.csv(paste0(param_dir, loc_group_file_name)))
  loc_grp$latitude = as.numeric(loc_grp$latitude)
  loc_grp$longitude = as.numeric(loc_grp$longitude)

  data$CountyGroup = 0L

  for(i in 1:nrow(loc_grp)) {
    data[latitude == loc_grp[i, latitude] & longitude == loc_grp[i, longitude], ]$CountyGroup = loc_grp[i, locationGroup]
  }
  return (data) 
}
##################################
#         vertdd function
##################################

generate_vertdd <- function(input_dir, file_name, 
                            lower_temp = 4.5, 
                            upper_temp=24.28){
  file_name <- paste0(input_dir, file_name, ".rds")
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

  data = data[, vert_Cum_dd := cumsum(vertdd), by=list(latitude, longitude, ClimateScenario, ClimateGroup, year)]
  data$vert_Cum_dd_F = data$vert_Cum_dd * 1.8

  data$cripps_pink = pnorm(data$vert_Cum_dd_F, mean = 495.51, sd = 42.58, lower.tail = TRUE)
  data$gala = pnorm(data$vert_Cum_dd_F, mean = 528.56, sd = 41.95, lower.tail = TRUE)
  data$red_deli = pnorm(data$vert_Cum_dd_F, mean = 522.74, sd = 42.79, lower.tail = TRUE)
  return(data)
}

############################################
bloom <- function(data){
  data = subset(data, select = c("ClimateGroup", 
                               "latitude", "longitude", 
                               "County", "ClimateScenario", 
                               "year", "month", "day", "dayofyear", 
                               "cripps_pink", "gala", "red_deli"))

  data = melt(data, id.vars = c("ClimateGroup", "latitude", "longitude", 
                              "County", "ClimateScenario", "year", "month", "day", "dayofyear"), 
                    variable.name = "apple_type")
  
  data = data[value >= 1.000000e+00,]
  data = data[, head(.SD, 1), by = c("ClimateGroup","latitude", "longitude","County","ClimateScenario","year", "apple_type")]
  data = data[, .(medDoY = as.integer(median(dayofyear))), by = c("ClimateGroup","latitude", "longitude","County", "apple_type")]
  return (data)
}




##################################
# Generation Generator
##################################
generations_func <- function(input_dir, file_name){
  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  generations  <- data[data[, month==8 & day==23]]
  generations1 <- data[data[, month==11 & day==5]]

  generations$NumAdultGens <- generations$PercAdultGen1 + generations$PercAdultGen2 + generations$PercAdultGen3 + generations$PercAdultGen4
  generations$NumLarvaGens <- generations$PercLarvaGen1 + generations$PercLarvaGen2 + generations$PercLarvaGen3 + generations$PercLarvaGen4

  generations1$NumAdultGens <- generations1$PercAdultGen1 + generations1$PercAdultGen2 + generations1$PercAdultGen3 + generations1$PercAdultGen4
  generations1$NumLarvaGens <- generations1$PercLarvaGen1 + generations1$PercLarvaGen2 + generations1$PercLarvaGen3 + generations1$PercLarvaGen4
  return (list(generations, generations1))
}


##################################
# Diapause Function
##################################

diapause_abs_rel <- function(input_dir, file_name,
                             param_dir, location_group_name="LocationGroups.csv"){
    file_name = paste0(input_dir, file_name, ".rds")
    data <- data.table(readRDS(file_name))

    loc_grp = data.table(read.csv(paste0(param_dir, location_group_name)))
    #loc_grp = loc_grp[1:15,]
    loc_grp$latitude = as.numeric(loc_grp$latitude)
    loc_grp$longitude = as.numeric(loc_grp$longitude)

    data <- data[latitude %in% loc_grp$latitude & longitude %in% loc_grp$longitude]

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
    rm (data)
    startingpopulationfortheyear<-1000
    #generation1
    sub[,LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), 
         by =list(year,ClimateScenario, 
         latitude,longitude,ClimateGroup, CountyGroup) ]
    sub$AbsPopLarvaGen1<-sub$LarvaGen1RelFraction*startingpopulationfortheyear
    sub$AbsPopLarvaGen1Diap<-sub$AbsPopLarvaGen1*sub$diapause1/100
    sub$AbsPopLarvaGen1NonDiap<-sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

    #generation2
    sub[,LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), 
         by = list(year, ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]
    sub[,AbsPopLarvaGen2 := LarvaGen2RelFraction * sum(AbsPopLarvaGen1NonDiap)*3.9, 
         by = list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]
    sub$AbsPopLarvaGen2Diap <- sub$AbsPopLarvaGen2 * sub$diapause1/100
    sub$AbsPopLarvaGen2NonDiap <- sub$AbsPopLarvaGen2 - sub$AbsPopLarvaGen2Diap

    #generation3
    sub[,LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), 
         by =list(year,ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]

    sub[,AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]
    sub$AbsPopLarvaGen3Diap <- sub$AbsPopLarvaGen3*sub$diapause1/100
    sub$AbsPopLarvaGen3NonDiap <- sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

    #generation4
    sub[,LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

    sub[,AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

    sub$AbsPopLarvaGen4Diap <- sub$AbsPopLarvaGen4*sub$diapause1/100
    sub$AbsPopLarvaGen4NonDiap <- sub$AbsPopLarvaGen4- sub$AbsPopLarvaGen4Diap

    ### get totals similar to Sum Larva column , but abs numbers
    sub$AbsPopTotal <- sub$AbsPopLarvaGen1 + sub$AbsPopLarvaGen2 + sub$AbsPopLarvaGen3 + sub$AbsPopLarvaGen4
    sub$AbsPopDiap <- sub$AbsPopLarvaGen1Diap + sub$AbsPopLarvaGen2Diap + sub$AbsPopLarvaGen3Diap + sub$AbsPopLarvaGen4Diap
    sub$AbsPopNonDiap <- sub$AbsPopLarvaGen1NonDiap + sub$AbsPopLarvaGen2NonDiap + 
                        sub$AbsPopLarvaGen3NonDiap + sub$AbsPopLarvaGen4NonDiap

    sub1 = subset(sub, 
                select = c("latitude", "longitude", 
                           "County", "CountyGroup", 
                           "ClimateScenario", "ClimateGroup", 
                           "year", "dayofyear", "CumDDinF", 
                           "SumLarva", "enterDiap", 
                           "escapeDiap", "AbsPopTotal",
                           "AbsPopNonDiap","AbsPopDiap"))

    sub1 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), 
                    RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), 
                    AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), 
                    CumulativeDDF = mean(CumDDinF)), 
                    by = c("ClimateGroup", "CountyGroup", "dayofyear")]
    
    #sub1 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), CumulativeDDF = mean(CumDDinF)), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude", "dayofyear")]

    RelData = subset(sub1, select = c("ClimateGroup", "CountyGroup", 
                                      "dayofyear", "CumulativeDDF", 
                                      "RelLarvaPop", "RelDiap", "RelNonDiap"))

    RelData = melt(RelData, id = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF"))
    #RelData[, Group := as.character()]
    RelData$Group = "0"
    temp1 = RelData[variable %in% c("RelLarvaPop", "RelDiap")]
    temp2 = RelData[variable %in% c("RelLarvaPop", "RelNonDiap")]
    temp1$Group = "Relative Larva Pop Vs Diapaused"
    temp2$Group = "Relative Larva Pop Vs NonDiapaused"
    RelData = rbind(temp1, temp2)

    AbsData = subset(sub1, select = c("ClimateGroup", "CountyGroup", 
                                      "dayofyear", "CumulativeDDF", 
                                      "AbsLarvaPop", "AbsDiap", "AbsNonDiap"))
    
    AbsData = melt(AbsData, id = c("ClimateGroup", "CountyGroup", "dayofyear", "CumulativeDDF"))
    #AbsData[, Group := as.character()]
    AbsData$Group = "0"
    temp1 = AbsData[variable %in% c("AbsLarvaPop", "AbsDiap")]
    temp2 = AbsData[variable %in% c("AbsLarvaPop", "AbsNonDiap")]
    temp1$Group = "Absolute Larva Pop Vs Diapaused"
    temp2$Group = "Absolute Larva Pop Vs NonDiapaused"
    AbsData = rbind(temp1, temp2)

    return (list(RelData, AbsData, sub1))
}
################################################################################
#    Generate Map 1
################################################################################

generate_diapause_map1 <- function(input_dir, file_name, 
                           param_dir, 
                           CodMothParams_name="CodlingMothparameters.txt", 
                           location_group_name="LocationGroups.csv"){
  CodMothParams <- read.table(paste0(param_dir, CodMothParams_name), header=TRUE, sep=",")
  sub1 = diapause_map1_prep(input_dir, file_name, param_dir, location_group_name="LocationGroups.csv")
  
  sub2 = sub1[, .(RelPctDiap = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100, RelPctNonDiap = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100, AbsPctDiap = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100, AbsPctNonDiap = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]
  
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctDiapGen1 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctDiapGen2 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctDiapGen3 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctDiapGen4 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctNonDiapGen1 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctNonDiapGen2 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctNonDiapGen3 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctNonDiapGen4 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  #
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctDiapGen1 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctDiapGen2 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctDiapGen3 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctDiapGen4 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  #
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctNonDiapGen1 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctNonDiapGen2 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctNonDiapGen3 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctNonDiapGen4 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
  return (sub2)
}

diapause_map1_prep <- function(input_dir, file_name,
                               param_dir, location_group_name="LocationGroups.csv"){
  file_name = paste0(input_dir, file_name, ".rds")
  data <- data.table(readRDS(file_name))

  loc_grp = data.table(read.csv(paste0(param_dir, location_group_name)))
  loc_grp$latitude = as.numeric(loc_grp$latitude)
  loc_grp$longitude = as.numeric(loc_grp$longitude)

  data <- data[latitude %in% loc_grp$latitude & longitude %in% loc_grp$longitude]

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
  
  #generation1
  sub[,LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), 
       by =list(year,ClimateScenario, 
       latitude,longitude,ClimateGroup, CountyGroup) ]
  sub$AbsPopLarvaGen1<-sub$LarvaGen1RelFraction*startingpopulationfortheyear
  sub$AbsPopLarvaGen1Diap<-sub$AbsPopLarvaGen1*sub$diapause1/100
  sub$AbsPopLarvaGen1NonDiap<-sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

  #generation2
  sub[,LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), 
       by = list(year, ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]
  sub[,AbsPopLarvaGen2 := LarvaGen2RelFraction * sum(AbsPopLarvaGen1NonDiap)*3.9, 
       by = list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]
  sub$AbsPopLarvaGen2Diap <- sub$AbsPopLarvaGen2 * sub$diapause1/100
  sub$AbsPopLarvaGen2NonDiap <- sub$AbsPopLarvaGen2 - sub$AbsPopLarvaGen2Diap

  #generation3
  sub[,LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), 
       by =list(year,ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]

  sub[,AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, 
       by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]
  sub$AbsPopLarvaGen3Diap <- sub$AbsPopLarvaGen3*sub$diapause1/100
  sub$AbsPopLarvaGen3NonDiap <- sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

  #generation4
  sub[,LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), 
       by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

  sub[,AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, 
       by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

  sub$AbsPopLarvaGen4Diap <- sub$AbsPopLarvaGen4*sub$diapause1/100
  sub$AbsPopLarvaGen4NonDiap <- sub$AbsPopLarvaGen4- sub$AbsPopLarvaGen4Diap

  ### get totals similar to Sum Larva column, but abs numbers
  sub$AbsPopTotal <- sub$AbsPopLarvaGen1 + sub$AbsPopLarvaGen2 + sub$AbsPopLarvaGen3 + sub$AbsPopLarvaGen4
  sub$AbsPopDiap <- sub$AbsPopLarvaGen1Diap + sub$AbsPopLarvaGen2Diap + sub$AbsPopLarvaGen3Diap + sub$AbsPopLarvaGen4Diap
  sub$AbsPopNonDiap <- sub$AbsPopLarvaGen1NonDiap + sub$AbsPopLarvaGen2NonDiap + 
                      sub$AbsPopLarvaGen3NonDiap + sub$AbsPopLarvaGen4NonDiap

  sub = subset(sub, select = c("latitude", "longitude", 
                               "County", "CountyGroup", 
                               "ClimateScenario", "ClimateGroup", 
                               "year", "dayofyear", "CumDDinF", 
                               "SumLarva", "enterDiap", 
                               "escapeDiap", "AbsPopTotal",
                               "AbsPopNonDiap","AbsPopDiap"))

  sub = sub[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), 
                                              RelNonDiap = mean(escapeDiap), 
                                              AbsLarvaPop = mean(AbsPopTotal), 
                                              AbsDiap = mean(AbsPopDiap), 
                                              AbsNonDiap = mean(AbsPopNonDiap), 
                                              CumulativeDDF = mean(CumDDinF)), 
                                              by = c("ClimateGroup", "CountyGroup", 
                                                      "latitude", "longitude", 
                                                      "dayofyear")]
  return (sub)
}



compute_cumdd_eggHatch <- function(input_dir, file_name="combined_CMPOP_", version){
  ## This is fir cumdd of egg hatch
  ##
  filename <- paste0(input_dir, file_name, version, ".rds")
  data <- data.table(readRDS(filename))
  data <- subset(data, select = c("CountyGroup", "latitude", "longitude", 
                                  "ClimateScenario", "ClimateGroup", 
                                  "year", "dayofyear", 
                                  "PercLarvaGen1", "PercLarvaGen2", 
                                  "PercLarvaGen3", "PercLarvaGen4"))
    
  data <- data[, .(LarvaGen1 = mean(PercLarvaGen1), LarvaGen2 = mean(PercLarvaGen2), 
                   LarvaGen3 = mean(PercLarvaGen3), LarvaGen4 = mean(PercLarvaGen4)), 
                    by = c("CountyGroup", "latitude", "longitude", "ClimateScenario", "ClimateGroup", "year", "dayofyear")]
    return (data)
}


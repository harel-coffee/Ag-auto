#.libPaths("/data/hydro/R_libs35")
.libPaths(c( .libPaths(), "~/R/lib"))
.libPaths()


library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
#library(ggplot2)
library(foreign)


options(digits=9)

# data_type [model scenario] data_from [data_to]
args = commandArgs(trailingOnly = TRUE)

root_data_dir <- '/data/adam/data/metdata/'
root_output_dir <- '/data/rajagopalan/CH_output_data/'

if (args[1] == 'obs') {
  data_dir <- paste0(root_data_dir, 'historical/UI_historical/VIC_Binary_CONUS_to_2016/')
  output_dir <- paste0(root_output_dir, 'modeled/obs_hist/')
  output_file_prefix <- 'CH_observed_historical_'
  hist <- T
  n_vars <- 8
  data_from_ix <- 2
  first_year <- 1979
  last_year <- 2016
} else if (args[1] == 'mod') {
  data_dir <- paste0(root_data_dir, 'maca_v2_vic_binary/', args[2], '/', args[3], '/')
  output_dir <- paste0(root_output_dir, 'modeled/', args[2], '/', args[3], '/')
  output_file_prefix <- 'CH_'
  hist <- (args[3] == 'historical')
  n_vars <- 4
  data_from_ix <- 4
  first_year <- 2006
  last_year <- 2099
  if (!file.exists(data_dir)) {
    err_msg <- sprintf('NO DATA AT %s', data_dir)
    print(err_msg)
    stop(err_msg)
  }
} else {
  err_msg <- sprintf('UNKNOWN DATA TYPE %s', args[1])
  print(err_msg)
  stop(err_msg)
}
dir.create(output_dir, recursive = T)
consolidated_path <- paste0(output_dir, 'consolidated.csv')

source_path <- '/data/rajagopalan/vlads_scripts_and_data/Cold/hardiness_core.R'
source(source_path)

param_dir <- '/data/rajagopalan/vlads_scripts_and_data/Cold/parameters/'

# read parameters
input_params  = data.table(read.csv(paste0(param_dir, "input_parameters", ".csv")))
variety_params = data.table(read.csv(paste0(param_dir, "variety_parameters", ".csv")))
all_data_locations = data.table(read.csv(paste0(param_dir,"data_locations",".csv")))
all_data_locations$ARRAYID <- as.integer(all_data_locations$ARRAYID)
if (length(args) > data_from_ix) {
  data_locations <- all_data_locations$Location[
    all_data_locations$ARRAYID >= as.integer(args[data_from_ix]) & 
      all_data_locations$ARRAYID <= as.integer(args[data_from_ix + 1])]
} else
  data_locations <- all_data_locations$Location[
    all_data_locations$ARRAYID == as.integer(args[data_from_ix])]
print(args)
print(data_locations)

for (data_location in data_locations) {
  filename = paste0(data_location)
  
  file_found <- list.files(path = data_dir, pattern = filename)
  file_found <- paste0(data_dir, filename)
  print(paste0('FOUND FILE: ', file_found))
  
  meta_data <- data.table(read_binary(file_path = file_found, hist = hist , no_vars = n_vars))
  # To calculate the jday, we combined the column of day, month and year and then converted it into date format, then into jday format and then to numeric
  # It was converted to numeric because the "%j" format is from 1-366, we needed 0-365, hence subtracting 1 from the number obtained
  
  meta_data <- meta_data %>% mutate(Date = paste0(meta_data$month,"/",meta_data$day,"/",meta_data$year))
  meta_data <- meta_data %>% mutate(jday = as.numeric(format(as.Date(meta_data$Date,format = "%m/%d/%y"),"%j"))-1) 
  meta_data <- meta_data %>% mutate(T_mean = (meta_data$tmax + meta_data$tmin)/2)
  meta_data <- meta_data %>% select("Date", "year", "jday", "T_mean", "tmax", "tmin")

  sapply(meta_data,class)
  
  start_time <- Sys.time()
  output <- hardiness_model(data = meta_data, input_params = input_params, variety_params = variety_params)
  end_time <- Sys.time()
  time_taken <- end_time - start_time
  print(paste0('COMPUTED IN: ', time_taken))
  # print(time_taken)
  
  output$hardiness_year <- ifelse(leap_year(output$year),ifelse(output$jday <=136,output$year-1,
                                                                ifelse(output$jday >=244 & output$jday < 367,output$year,0)),
                                  ifelse(output$jday <=135,output$year-1,
                                         ifelse(output$jday >=243 & output$jday < 366,output$year,0)))
  
  output_CDI = data.table(matrix(NA, nrow =1, ncol = last_year - first_year + 1))
  colnames(output_CDI) <- paste(first_year:last_year,sep =" ")
  # find the count anamolies per hardiness year
  for (i in colnames(output_CDI))
  {
    output_CDI[[i]] = sum(subset(output, hardiness_year == i)$CDI)
  }
  
  output_CDI <- cbind(Time_elapsed = time_taken, output_CDI )
  output_CDI <- cbind(Location = data_location,output_CDI)
  
  ######################################### OUTPUT
  #write.csv(output, file = paste0(output_dir, "CH_", filename,".csv"), row.names=FALSE)
  output_name <- paste0(output_dir, output_file_prefix, filename, ".csv")
  write.table(output, file=output_name, row.names=F, sep=',', append=T, col.names=!file.exists(output_name))
  
  write.table(output_CDI, consolidated_path, sep = ',', col.names = !file.exists(consolidated_path),
              row.names = FALSE, append = TRUE)  
}
# !! uncomment if needed?
# write.table(output_CDI, file = paste0(output_dir, "consolidated_observed_historical.csv"),sep = ",",
#             col.names = !file.exists(paste0(output_dir, "consolidated_observed_historical.csv", ".csv")),
#             row.names= FALSE,append = TRUE )



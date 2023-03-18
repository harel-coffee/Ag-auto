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
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source_2 = "/home/hnoorazar/reading_binary/read_binary_core.R"
source(source_1)
source(source_2)
options(digits=9)
options(digit=9)

######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################
#
# Define main output path
#
bloom_out = "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/"
main_out <- file.path(bloom_out, "/01_modeled_bloom/")

######################################################################
##                                                                  ##
##              command_line_arguments                              ##
##                                                                  ##
######################################################################
# args = commandArgs(trailingOnly=TRUE)

# dt_mu = args[1]   # w_precip # no_recip
# st_date = args[2]
######################################################################

# 2. Pre-processing prep -----------------------------------------
# 2a. Only use files in geographic locations we're interested in

param_dir = file.path("/home/hnoorazar/bloom_codes/parameters/")
# local_files <- read.delim(file = paste0(param_dir, 
#                                         "file_list.txt"), 
#                           header=FALSE, as.is=TRUE)
# local_files <- as.vector(local_files$V1)

local_files <- c("data_48.40625_-119.53125", "data_46.59375_-120.53125",
                 "data_46.03125_-118.34375", "data_44.03125_-123.09375")

# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern = "historical", x = getwd()) == T, TRUE, FALSE)

# Get current folder
pp <- "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/"
current_dir <- gsub(x = getwd(),
                    pattern = pp,
                    replacement = "")

current_model <- gsub("-", "_", basename(dirname(current_dir)))
current_emission <- basename(current_dir)

print("does this look right?")
print(file.path(main_out, current_dir))
print (paste0("current_dir is ", current_dir))
print (paste0("model is ", current_model))
print (paste0("emission is ", current_emission))

if (dir.exists(file.path(main_out, current_dir)) == F){
  dir.create(path=file.path(main_out, current_dir), recursive=T)
}

# get files in current folder and remove non-data files
dir_con <- dir()
dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

start_time <- Sys.time()

##############################
######
######  Run the code
######
##############################

st_dates <- c(1, 7, 14, 21, 28, 35, 42)
dist_means <- c(327, 348.8, 370.6, 392.4, 414.2, 436, 457.8, 479.6, 501.4, 523.2, 545)

for (st_date in st_dates){
  for (dt_mu in dist_means){
    for(file in dir_con){

      # 3a. read in binary meteorological data file from specified path
      
      met_data <- read_binary(file_path=file, hist=hist, no_vars=4)

      # I make the assumption that lat always has 
      # same number of decimal points
      lat <- as.numeric(substr(x=file, start=6, stop=13))
      long <- as.numeric(substr(x=file, start=15, stop=25))
      met_data$lat <- lat
      met_data$long <- long

      # 3b. Clean it up
      met_data <- within(met_data, remove(precip, windspeed)) %>%
                  data.table()
      met_data$model <- current_model

      print ("line 96 of modeled")  
      if (current_emission=="rcp85"){
         met_data$emission <- "RCP 8.5"
         } else if (current_emission=="rcp45"){
          met_data$emission <- "RCP 4.5"
         } else {
          met_data$emission <- "modeled_hist"
      }
      
      # 3c. get vertical DD.
      met_data <- generate_vertdd_for_sensitivity(data_tb = met_data, 
                                                  lower_temp = 4.5, 
                                                  upper_temp = 24.28,
                                                  distribution_mean = dt_mu,
                                                  start_doy = st_date)

      met_data$start_accum_date <- st_date
      met_data$dist_mean <- dt_mu
      
      met_data$location <- paste0(met_data$lat, "_",met_data$long)
      met_data <- within(met_data, remove(tmax, tmin, lat, long))
      met_data <- data.table(met_data)

      # met_data <- put_chill_calendar(met_data, chill_start="sept")
      # met_data <- add_chill_sept_DoY(met_data)
      
      # 3d. Save output
      
      output_name <- paste0("/bloom_", lat, "_", long, "_start_Jan_" , st_date, "_NormalMean_", dt_mu, ".rds")
      saveRDS(object=met_data,
              file=file.path(main_out, current_dir, output_name))

      rm(met_data)
    }
  }
}




# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)





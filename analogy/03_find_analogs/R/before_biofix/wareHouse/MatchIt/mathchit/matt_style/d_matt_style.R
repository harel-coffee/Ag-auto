.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(MESS) # has the auc function in it.
library(geepack)
library(chron)
library(MatchIt)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)
#######################################################################
#############################
#                           #
#       shell arguments     #
#                           #
#############################

# args = commandArgs(trailingOnly=TRUE)

# model_name <- args[1]
# carbon_type <- args[2]
#############################
#                           #
#    matching parameters    #
#                           #
#############################
m_method <- "nearest"
m_distance <- "mahalanobis"
m_ratio <- 700
precip <- TRUE
#######################################################################
#######################################################################
##########################################################
#                                                        #
#                      directories                       #
#                                                        #
##########################################################

main_out <- "/data/hydro/users/Hossein/analog/z_R_results"

print("does this look right? This is where we are now")
getwd()
print ("_______________________________________________")
print ("main_out is ")
print (main_out)
print ("_______________________________________________")

curr_out <- file.path(main_out, m_method, m_distance)
print ("curr_out is (to be completed)")
print (curr_out)
print ("_______________________________________________")
print ("line 55, critical")
print (getwd())
curr_sub <- gsub(x = getwd(),
                 pattern = "/data/hydro/users/Hossein/analog/local/ready_features/broken_down_location_level_coarse/rcp85/",
                 replacement = "")
print ("curr_sub is ")
print (curr_sub)
print ("_______________________________________________")

final_out <- file.path(curr_out, curr_sub)
print ("This is where the output will be going (final_out) ")
print (final_out)

#######################################################################
if (dir.exists(file.path(final_out)) == F){
  dir.create(path = final_out, recursive = T)
}
#######################################################################

ctrl_data <- data.table(readRDS("/data/hydro/users/Hossein/analog/usa/ready_features/all_data_usa.rds"))

# current one single file corresponding to a given
# location and year to be read as base file.
dir_con <- dir()
print ("_________________*************_________________")
print (getwd())
print ("_________________*************________________")
# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "feat_",
                        x = dir_con)]

print ("There should be one file here, is it?")
print (dir_con)
print ("_______________________________________________")
counter = 0
start_time <- Sys.time()
for (file in dir_con){
  all_loc_data <- data.table(readRDS(file))
  years = unique(all_loc_data$year)

  for (yr in years){
    print (yr)
    start_time_one_run <- Sys.time()
    counter = counter + 1
    base <- all_loc_data %>% filter(year == yr)
    print (dim(base))
    match_out <- sort_matchit_out(base=base, usa=ctrl_data, 
                                  m_method=m_method, 
                                  m_distance=m_distance, 
                                  m_ratio=m_ratio, 
                                  precip=precip)
    saveRDS(match_out, paste0(final_out, "/", base$location, "_", base$year, ".rds"))
    if (counter == 1){print (Sys.time() - start_time_one_run)}
    
  }
}

end_time <- Sys.time()
print(end_time - start_time)



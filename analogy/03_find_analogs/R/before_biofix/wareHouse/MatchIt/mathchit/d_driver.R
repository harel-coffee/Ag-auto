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

args = commandArgs(trailingOnly=TRUE)

# model_name <- args[1]
# carbon_type <- args[2]
file_name <- args[1]

# m_method <- args[1]    # method for matching, e.g. nearest, optimal, full
# m_distance <- args[2]  # the metric to be used for matchit()
# m_ratio <- as.numeric(args[3])     # number of locations to be matched with a given site
# precip <- args[3]

m_method <- "nearest"
m_distance <- "Mahalanobis"
m_ratio <- 700
precip <- TRUE

#######################################################################
##########################################################
#                                                        #
#                      directories                       #
#                                                        #
##########################################################

param_dir <- "/home/hnoorazar/cleaner_codes/parameters/"
in_dir <- "/data/hydro/users/Hossein/analog/local/ready_features/"

main_out <- "/data/hydro/users/Hossein/analog/z_R_results/"
curr_out <- file.path(main_out, m_method, m_distance)

if (dir.exists(file.path(curr_out)) == F){
  dir.create(path = curr_out, recursive = T)
}
#######################################################################

ctrl_data <- data.table(readRDS("/data/hydro/users/Hossein/analog/usa/ready_features/all_data_usa.rds"))
base <- data.table(readRDS(paste0(in_dir, model_carbon, "/", file_name)))

start_time <- Sys.time()

match_out <- sort_matchit_out(base=base, usa=ctrl_data, 
                              m_method=m_method, 
                              m_distance=m_distance, 
                              m_ratio=m_ratio, 
                              precip=precip)

saveRDS(match_out, paste0(curr_out, "/", base$location, "_", base$year, ".rds"))

end_time <- Sys.time()
print(end_time - start_time)



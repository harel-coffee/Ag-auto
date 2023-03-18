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

m_method <- args[1]    # method for matching, e.g. nearest, optimal, full
m_distance <- args[2]  # the metric to be used for matchit()
m_ratio <- as.numeric(args[3])     # number of locations to be matched with a given site
precip <- args[4]
#######################################################################
#############################
#                           #
#       directories         #
#                           #
#############################

param_dir <- "/home/hnoorazar/cleaner_codes/parameters/"
in_dir <- "/data/hydro/users/Hossein/analog/test/"

main_out <- "/data/hydro/users/Hossein/analog/test/results/"
curr_out = file.path(main_out, m_method, m_distance)

if (dir.exists(file.path(curr_out)) == F){
  dir.create(path = curr_out, recursive = T)
}
#######################################################################


treat_data <- data.table(readRDS(paste0(in_dir, "averaged_data_rcp85.rds")))
ctrl_data <- data.table(readRDS(paste0(in_dir, "all_data_usa.rds")))

start_time <- Sys.time()

for (ii in 1:dim(treat_data)[1]){
    base <- treat_data[ii, ]
    if (ii==1){start_one_ex <- Sys.time()}
    match_out <- sort_matchit_out(base=base, usa=ctrl_data, 
                                  m_method=m_method, 
                                  m_distance=m_distance, 
                                  m_ratio=m_ratio, 
                                  precip=precip)
    
    saveRDS(match_out, paste0(curr_out, "/", base$location, "_", base$year, ".rds"))
    
    if (ii==1){
      print (paste0("one example takes the following amount of time: ", (Sys.time() - start_one_ex)))
    }
}
end_time <- Sys.time()
print(end_time - start_time)



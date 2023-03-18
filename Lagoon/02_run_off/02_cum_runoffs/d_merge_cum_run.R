###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                      Define all paths                            ##
##                                                                  ##
######################################################################
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

data_dir <- "/data/hydro/users/Hossein/lagoon/02_run_off/01_cum_runs/"

annual_in <- paste0(data_dir, "annual/")
chunky_in <- paste0(data_dir, "chunk/")
monthly_in <- paste0(data_dir, "month/")
wtr_yr_in <- paste0(data_dir, "wtr_yr/")

out_dir <- data_dir
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}

######################################################################
##                                                                  ##
##                                                                  ##
##                                                                  ##
######################################################################

######################################
#
#       annual
#
######################################
setwd(annual_in)
getwd()
the_dir <- dir(annual_in, pattern = ".rds")

all_ann <- data.table()

for (file in the_dir){
  all_ann <- rbind(all_ann, data.table(readRDS(file)))
}
all_ann <- within(all_ann, remove(cluster.x))

if ("cluster.y" %in% colnames(all_ann)){
  setnames(all_ann, old=c("cluster.y"), new=c("cluster"))
}

saveRDS(all_ann, paste0(out_dir, "/all_ann_cum_runbase.rds"))
rm(all_ann)
######################################
#
#       chunky
#
######################################
setwd(chunky_in)
getwd()
the_dir <- dir(chunky_in, pattern = ".rds")

all_chunk <- data.table()

for (file in the_dir){
  all_chunk <- rbind(all_chunk, data.table(readRDS(file)))
}
all_chunk <- within(all_chunk, remove(cluster.x))

if ("cluster.y" %in% colnames(all_chunk)){
  setnames(all_chunk, old=c("cluster.y"), new=c("cluster"))
}

saveRDS(all_chunk, paste0(out_dir, "/all_chunk_cum_runbase.rds"))
rm(all_chunk)
######################################
#
#       Water Year
#
######################################
setwd(wtr_yr_in)
getwd()
the_dir <- dir(wtr_yr_in, pattern = ".rds")

all_wtr_yr <- data.table()

for (file in the_dir){
  all_wtr_yr <- rbind(all_wtr_yr, data.table(readRDS(file)))
}
all_wtr_yr <- within(all_wtr_yr, remove(cluster.x))

if ("cluster.y" %in% colnames(all_wtr_yr)){
  setnames(all_wtr_yr, old=c("cluster.y"), new=c("cluster"))
}

saveRDS(all_wtr_yr, paste0(out_dir, "/all_wtr_yr_cum_runbase.rds"))
rm(all_wtr_yr)

######################################################################
# monthly
#
setwd(monthly_in)
getwd()
the_dir <- dir(monthly_in, pattern = ".rds")

all_monthly <- data.table()

for (file in the_dir){
  all_monthly <- rbind(all_monthly, data.table(readRDS(file)))
}

suppressWarnings({ all_monthly <- within(all_monthly, remove(cluster.x))})
if ("cluster.y" %in% colnames(all_monthly)){
  setnames(all_monthly, old=c("cluster.y"), new=c("cluster"))
}

saveRDS(all_monthly, paste0(out_dir, "/all_monthly_cum_runbase.rds"))
rm(all_monthly)

##################################################
end_time <- Sys.time()
print( end_time - start_time)
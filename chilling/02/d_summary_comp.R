
.libPaths("/data/hydro/R_libs35")
.libPaths()

#####################################################
###                                               ###
###             Sept. thru Dec and Jan            ###
###                                               ###
#####################################################

rm(list=ls())
library(data.table)
library(dplyr)
options(digit=9)
options(digits=9)

in_pref <- "/data/hydro/users/Hossein/chill/data_by_core/dynamic/02/"
direcs <- c("sept", "mid_sept", 
            "oct", "mid_oct", 
            "nov", "mid_nov")

starts <- c("Sept. 1", "Sept. 15", 
            "Oct. 1",  "Oct. 15", 
            "Nov. 1",  "Nov. 15")

for (i in 1:length(direcs)){
  specific_dir <- direcs[i]
  start = starts[i]
  
  write_dir <- file.path(in_pref, specific_dir)
  in_dir <- file.path(write_dir, "/non_overlap/")
  print(specific_dir)
  print(in_dir)
  
  setwd(in_dir)
  the_dir <- dir(in_dir, pattern = ".txt")

  # remove filenames that aren't data
  the_dir <- the_dir[grep(pattern = "summary", x = the_dir)]

  # Compile the data files for plotting
  summary_comp <- lapply(the_dir, read.table, header=T)
  summary_comp <- do.call(bind_rows, summary_comp)
  print(sort(colnames(summary_comp)))

  summary_comp <- within(summary_comp, remove(.id))
  summary_comp <- na.omit(summary_comp)

  summary_comp$start <- start
  summary_comp$location <- paste0(summary_comp$lat, "_", 
                                  summary_comp$long)
  print(unique(summary_comp$time_period))
  saveRDS(summary_comp, paste0(write_dir, "/", 
                               specific_dir, 
                               "_summary_comp.rds"))
}

main_in <- "/data/hydro/users/Hossein/chill/data_by_core/dynamic/02/"

mid_nov <- data.table(readRDS(paste0(main_in, 
                              "mid_nov/mid_nov_summary_comp.rds")))

mid_oct <- data.table(readRDS(paste0(main_in, 
                              "mid_oct/mid_oct_summary_comp.rds")))

mid_sept<- data.table(readRDS(paste0(main_in, 
                              "mid_sept/mid_sept_summary_comp.rds")))

nov <- data.table(readRDS(paste0(main_in, 
                          "/nov/nov_summary_comp.rds")))

oct <- data.table(readRDS(paste0(main_in, 
                          "/oct/oct_summary_comp.rds")))

sept <-data.table(readRDS(paste0(main_in, 
                          "/sept/sept_summary_comp.rds")))

all_summary_comps <- rbind(mid_nov, mid_oct, mid_sept, nov, oct, sept)
saveRDS(all_summary_comps, paste0(main_in, "/all_summary_comps.rds"))


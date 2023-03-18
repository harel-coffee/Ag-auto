
.libPaths("/data/hydro/R_libs35")
.libPaths()

#####################################################
###                                               ###
###             Sept.                             ###
###                                               ###
#####################################################

rm(list=ls())
library(data.table)
library(dplyr)
options(digit=9)
options(digits=9)

in_dir <- "/data/hydro/users/Hossein/chill/data_by_core/02_Springer_data/"
modeled_dir <- file.path(in_dir, "/non_overlap/")
write_dir <- in_dir


start <- "Sept. 1"

print(modeled_dir)
setwd(modeled_dir)
the_dir <- dir(modeled_dir, pattern = ".txt")

# remove filenames that aren't data
the_dir <- the_dir[grep(pattern = "summary", x = the_dir)]

# Compile the data files for plotting
summary_comp <- lapply(the_dir, read.table, header=T)
summary_comp <- do.call(bind_rows, summary_comp)
print(sort(colnames(summary_comp)))

summary_comp <- within(summary_comp, remove(.id))
summary_comp <- na.omit(summary_comp)

summary_comp$start <- start
summary_comp$location <- paste0(summary_comp$lat, "_", summary_comp$long)


#____________________________________________________________
#
# observed
#
observed_dt <- read.table(file = paste0(in_dir, "summary_obs_hist_springer_lowVariety.txt"), header = T)
observed_dt <- within(observed_dt, remove(.id))
observed_dt$start <- start
observed_dt$location <- paste0(observed_dt$lat, "_", observed_dt$long)
observed_dt$time_period <- "Observed"
observed_dt$emission <- "Observed"

observed_dt <- data.table(observed_dt)
summary_comp <- data.table(summary_comp)

summary_comp <- rbind(observed_dt, summary_comp)

print(unique(summary_comp$time_period))
saveRDS(summary_comp, paste0(write_dir, "/",
                             "summary_comp_springer_lowVariety.rds"))



.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
##################################################################
##                                                              ##
##              Terminal/shell/bash arguments                   ##
##                                                              ##
##################################################################
#
# Define main output path
#
FB_out = "/data/hydro/users/Hossein/bloom/02_bloomCut_first_frost/"
main_out <- file.path(FB_out, "/frost/")
if (dir.exists(main_out) == F) {
  dir.create(path = main_out, recursive = T)
}

# 2b. Note if working with a directory of historical data
hist <- TRUE

# 2d. Prep list of files for processing

# get files in current folder
the_dir <- dir()
the_dir <- the_dir[grep(pattern = ".rds", x = the_dir)]
print(head(the_dir))

# 3. Process the data ------------------------------------------

all_obs_frost <- data.table()
for(file in the_dir){
  met_data <- data.table(readRDS(file))
  met_data <- trim_chill_calendar(met_data)
  first_frost <- find_1st_frost(met_data)
  all_obs_frost <- rbind(all_obs_frost, first_frost)
}
saveRDS(object=all_obs_frost,
        file=file.path(main_out, "/frost_observed.rds"))


end_time <- Sys.time()
print( end_time - start_time)



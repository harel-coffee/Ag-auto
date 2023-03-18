###################################################################
##
##
##   We want to pick up the cumDD or cumGDD on 
##   the days where 50% of bloom is reached. to plot a box plot
##   as a response to reviewers of the chill paper.
##
##
##
###################################################################
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
main_out = "/data/hydro/users/Hossein/bloom/05_GDD_at_50PercBloom_chillPaperReviewerResponse/"
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

all_fullbloom_50percent_day <- data.table()
for(file in the_dir){
  met_data <- data.table(readRDS(file))
  bloom_cuts <- cumGDD_cut_at_50PercBloom(data=met_data, cut_off=0.5)
  bloom_cuts <- put_chill_calendar(bloom_cuts, chill_start="sept")
  bloom_cuts <- convert_doy_to_chill_doy(bloom_cuts)
  bloom_cuts <- trim_chill_calendar(bloom_cuts)
  all_fullbloom_50percent_day <- rbind(all_fullbloom_50percent_day, 
                                       bloom_cuts)
}
saveRDS(object=all_fullbloom_50percent_day,
        file=paste0(main_out, "/fullbloom_50percent_day_observed_VertGDD.rds"))

end_time <- Sys.time()
print( end_time - start_time)


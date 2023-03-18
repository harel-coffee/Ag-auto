.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

###########################################################

main_in <- "/data/hydro/users/Hossein/lagoon/00_raw_data/"

###########################################################
fiels <- c("raw_modeled_hist.rds", "raw_observed.rds",
           "raw_RCP45.rds", "raw_RCP85.rds")

negative_precips <- data.table()

for (file in fiels){
  A <- data.table(readRDS(paste0(main_in, file)))
  A <- A %>%
       filter(precip < 0)
  negative_precips <- rbind(negative_precips, A)
}
print (dim(negative_precips))
saveRDS(negative_precips, paste0(main_in, "negative_precips.rds"))

end_time <- Sys.time()
print( end_time - start_time)



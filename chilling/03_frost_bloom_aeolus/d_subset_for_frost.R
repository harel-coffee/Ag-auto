
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digit=9)
options(digits=9)


main_in <- "/data/hydro/users/Hossein/codling_moth_new/local/processed/"

main_out <- "/data/hydro/users/Hossein/chill/frost_bloom_initian_database/"
if (dir.exists(file.path(main_out)) == F) {dir.create(path = main_out, recursive = T)}

for (emission in c("rcp45.rds", "rcp85.rds")){
  dt <- data.table(readRDS(paste0(main_in, "combined_CMPOP_", emission)))
  print (sort(colnames(dt)))
  dt <- subset(dt, select=c(latitude, longitude, tmin, tmax, year, month, dayofyear, day, ClimateScenario))
  print ("_______________________________")
  print ("line 21")
  print (sort(colnames(dt)))
  dt <- dt %>% filter(month %in% c(9, 10, 11, 12))
  dt <- dt %>% filter(tmin <= 0)
  saveRDS(dt, paste0(main_out, "dt_for_frost_", emission))
}

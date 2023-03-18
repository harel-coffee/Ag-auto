.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

# This fukcing function is created because unique, ~duplicate,
# nothing could work! So, we first separate the data
# then bind it together with another function.
# then compute diapause stuff
#

in_dir = "/data/hydro/users/Hossein/analog/local/data_bases/"
out_dir = "/data/hydro/users/Hossein/analog/local/data_bases/001_unique_CMPOP/"

CMPOP_2040_rcp45 <- data.table(readRDS(paste0(in_dir, "CMPOP_2040_rcp45.rds")))
CMPOP_2060_rcp45 <- data.table(readRDS(paste0(in_dir, "CMPOP_2060_rcp45.rds")))

CMPOP_rcp45 = rbind(CMPOP_2040_rcp45, CMPOP_2060_rcp45)
rm(CMPOP_2040_rcp45, CMPOP_2060_rcp45)

CMPOP_2080_rcp45 <- data.table(readRDS(paste0(in_dir, "CMPOP_2080_rcp45.rds")))

CMPOP_rcp45 = rbind(CMPOP_rcp45, CMPOP_2080_rcp45)
CMPOP_rcp45 = data.table(CMPOP_rcp45)

saveRDS(CMPOP_rcp45, paste0(out_dir, "CMPOP_rcp45.rds"))
rm(CMPOP_rcp45, CMPOP_2080_rcp45)

################ RCP 85

CMPOP_2040_rcp85 <- data.table(readRDS(paste0(in_dir, "CMPOP_2040_rcp85.rds")))
CMPOP_2060_rcp85 <- data.table(readRDS(paste0(in_dir, "CMPOP_2060_rcp85.rds")))

CMPOP_rcp85 = rbind(CMPOP_2040_rcp85, CMPOP_2060_rcp85)
rm(CMPOP_2040_rcp85, CMPOP_2060_rcp85)

CMPOP_2080_rcp85 <- data.table(readRDS(paste0(in_dir, "CMPOP_2080_rcp85.rds")))

CMPOP_rcp85 = rbind(CMPOP_rcp85, CMPOP_2080_rcp85)
CMPOP_rcp85 = data.table(CMPOP_rcp85)
saveRDS(CMPOP_rcp85, paste0(out_dir, "CMPOP_rcp85.rds"))
rm(CMPOP_rcp85, CMPOP_2080_rcp85)




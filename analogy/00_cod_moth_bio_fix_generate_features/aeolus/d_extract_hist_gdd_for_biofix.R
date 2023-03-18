
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(tidyverse)
library(lubridate)
library(dplyr)
library(data.table)

options(digit=9)
options(digits=9)

source_path = "/home/hnoorazar/analog_codes/00_biofix/core_codMoth_bio_fix.R"
source(source_path)

################################################################################
#******************               Read Carefully!            ******************#
#******************               Read Carefully!            ******************#
#******************               Read Carefully!            ******************#
#
# This is HISTORICAL data. 
#     It does not have RCP in it.
#     It does not have year 2047 in it, no worries about overlapping 
#
#
#
#################################################################################

st_time <- Sys.time()

data_dir_base <- "/data/hydro/users/Hossein/analog/usa/data_bases/"

data_dir <- paste0(data_dir_base, "before_biofix/")
combined_CMPOP <- data.table(readRDS(paste0(data_dir, "combined_CMPOP.rds")))
combined_CMPOP$location <- paste0(combined_CMPOP$latitude, "_", combined_CMPOP$longitude)

saveRDS(combined_CMPOP, paste0(data_dir, "combined_CMPOP.rds"))

print (sort(colnames(combined_CMPOP)))
print("_______________________________")
print ("unique(ClimateGroup) ")
print (unique(combined_CMPOP$ClimateGroup))

print("_______________________________")
print ("unique(ClimateScenario)")
print (unique(combined_CMPOP$ClimateScenario))

print("_______________________________")
print ("unique(CountyGroup)")
print (unique(combined_CMPOP$CountyGroup))

print("_______________________________")

needed_cols <- c("location", "year", "month", "day", "DailyDD", "CumDDinC")
bad_CMPOP <- subset(combined_CMPOP, select = needed_cols)

################################################################################
#******************                Apply biofix              ******************#
#******************                Apply biofix              ******************#
#******************                Apply biofix              ******************#

param_dir <- "/home/hnoorazar/analog_codes/parameters/"
biofix_param <- data.table(read.csv(paste0(param_dir, "biofix_param_hi.csv"), 
                                    header=T, as.is=T))

good_CMPOP <- apply_bio_fix_to_CMPOP(bad_CMPOP, biofix_param)
good_CMPOP$model <- "observed"
good_CMPOP <- data.table(good_CMPOP)

################################################################################
#******************         save in right directory          ******************#
#******************         save in right directory          ******************#
#******************         save in right directory          ******************#

out_dir <- paste0(data_dir_base, "biofixed/")
saveRDS(good_CMPOP, paste0(out_dir, "good_historical_CMPOP.rds"))


good_CMPOP <- good_CMPOP %>% filter(month == 8 & day==23) %>% data.table()
setnames(good_CMPOP, old=c("CumDDinC", "CumDDinF"), new=c("CumDDinC_Aug23", "CumDDinF_Aug23"))
saveRDS(good_CMPOP, paste0(out_dir, "good_historical_CMPOP_Aug23.rds"))

################################################################################
#******************         read precip and combine          ******************#
#******************         read precip and combine          ******************#
#******************         read precip and combine          ******************#

precip_usa <- data.table(readRDS(paste0(data_dir_base, "precip_usa.rds")))

#### To be done once (START)
# setnames(precip_usa, old=c("precip"), new=c("yearly_precip"))
# saveRDS(precip_usa, paste0(data_dir_base, "precip_usa.rds"))
#### To be done once (END)

precip_usa <- subset(precip_usa, select=c("location", "year", "yearly_precip"))
good_CMPOP <- subset(good_CMPOP, select=c("location", "year", "CumDDinF_Aug23"))

hist_CDD_precip <- merge(precip_usa, good_CMPOP, by=c("location", "year"))
hist_CDD_precip$model <- "observed"

new_col_order <- c("location", "year", "CumDDinF_Aug23", "yearly_precip", "model")
setcolorder(hist_CDD_precip, new_col_order)

bio_fixed_ready_deat_dir <- "/data/hydro/users/Hossein/analog/usa/ready_features/biofixed/"
saveRDS(hist_CDD_precip, paste0(bio_fixed_ready_deat_dir, "hist_CDD_precip.rds"))

print ("It took goddamn: ")
print (Sys.time() - st_time)





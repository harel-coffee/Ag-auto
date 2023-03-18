.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)
             
#####################################################################################
st_time <- Sys.time()

param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

input_dir = "/data/hydro/users/Hossein/analog/local/data_bases/before_biofix/001_unique_CMPOP/"
write_dir = "/data/hydro/users/Hossein/analog/local/data_bases/biofixed/"

CMPOP_rcp45 <- data.table(readRDS(paste0(input_dir, "CMPOP_rcp45.rds")))
CMPOP_rcp85 <- data.table(readRDS(paste0(input_dir, "CMPOP_rcp85.rds")))

print ("line 23")
print (sort(colnames(CMPOP_rcp45)))

needed_cols <- c("location", "year", "month", "day", "ClimateScenario", "CumDDinF")
CMPOP_rcp45 <- subset(CMPOP_rcp45, select = needed_cols)
CMPOP_rcp85 <- subset(CMPOP_rcp85, select = needed_cols)

print ("line 30")

CMPOP_rcp45 <- CMPOP_rcp45 %>% filter(month == 8 & day == 23)
CMPOP_rcp85 <- CMPOP_rcp85 %>% filter(month == 8 & day == 23)

setnames(CMPOP_rcp45, old=c("CumDDinF"), new=c("CumDDinF_Aug23"))
setnames(CMPOP_rcp85, old=c("CumDDinF"), new=c("CumDDinF_Aug23"))

saveRDS(CMPOP_rcp45, paste0(write_dir, "CMPOP_rcp45_Aug23.rds"))
saveRDS(CMPOP_rcp85, paste0(write_dir, "CMPOP_rcp85_Aug23.rds"))

################################################################################
#******************         read precip and combine          ******************#
#******************         read precip and combine          ******************#
#******************         read precip and combine          ******************#

precip_dir <- "/data/hydro/users/Hossein/analog/local/data_bases/before_biofix/"

precip_rcp45 <- data.table(readRDS(paste0(precip_dir, "precip_rcp45.rds")))
precip_rcp85 <- data.table(readRDS(paste0(precip_dir, "precip_rcp85.rds")))

print ("sort(colnames(precip_rcp45))")
print (sort(colnames(precip_rcp45)))
print ("_______________________________")

#### To be done once (START)
# setnames(precip_rcp45, old=c("precip"), new=c("yearly_precip"))
# setnames(precip_rcp85, old=c("precip"), new=c("yearly_precip"))

# saveRDS(precip_rcp45, paste0(precip_dir, "precip_rcp45.rds"))
# saveRDS(precip_rcp85, paste0(precip_dir, "precip_rcp85.rds"))
#### To be done once (END)

CMPOP_rcp45 <- data.table(CMPOP_rcp45)
CMPOP_rcp85 <- data.table(CMPOP_rcp85)

precip_rcp45 <- data.table(precip_rcp45)
precip_rcp85 <- data.table(precip_rcp85)

CDD_precip_rcp45 <- merge(CMPOP_rcp45, precip_rcp45, by=c("location", "year", "ClimateScenario"))
CDD_precip_rcp85 <- merge(CMPOP_rcp85, precip_rcp85, by=c("location", "year", "ClimateScenario"))

CDD_precip_rcp45 <- within(CDD_precip_rcp45, remove(month, day))
CDD_precip_rcp85 <- within(CDD_precip_rcp85, remove(month, day))

new_col_order <- c("location", "year", "CumDDinF_Aug23", "yearly_precip", "ClimateScenario")
setcolorder(CDD_precip_rcp45, new_col_order)
setcolorder(CDD_precip_rcp85, new_col_order)

setnames(CDD_precip_rcp45, old=c("ClimateScenario"), new=c("model"))
setnames(CDD_precip_rcp85, old=c("ClimateScenario"), new=c("model"))

feat_dir <- "/data/hydro/users/Hossein/analog/local/ready_features/biofixed/"
saveRDS(CDD_precip_rcp45, paste0(feat_dir, "CDD_precip_rcp45.rds"))
saveRDS(CDD_precip_rcp85, paste0(feat_dir, "CDD_precip_rcp85.rds"))


print ("It took goddamn: ")
print (Sys.time() - st_time)





.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(geepack)
library(chron)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)

options(digits=9)
options(digit=9)

param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
main_in_dir <- "/data/hydro/users/Hossein/analog/usa/data_bases/"
main_out_dir <- "/data/hydro/users/Hossein/analog/usa/ready_features/"

##################################################################
##
##            Terminal Arguments
##
##################################################################

##################################################################


#### Create First Flight Median Day of Year
FF_dt <- data.table(readRDS("/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/combined_CM.rds"))
FF_dt <- generate_mDoY_FF(FF_dt)
##
##
gen_dt <- data.table(readRDS(paste0(main_in_dir, "generations_Aug.rds")))
gen_dt$location <- paste0(gen_dt$latitude, "_", gen_dt$longitude)
gen_dt <- subset(gen_dt, select = c("year", "location", "NumLarvaGens"))

# the following change is being done so data have the same column names
setnames(gen_dt, old=c("NumLarvaGens"), new=c("NumLarvaGens_Aug"))

###
### Create relative fraction of escaped diapause
###

diapause_dt <- data.table(readRDS(paste0(main_in_dir, "diapause_map1_rel_observed.rds")))
###
### Read precip
###

precip_dt <- data.table(readRDS(paste0(main_in_dir, "precip_usa.rds")))
setnames(precip_dt, old=c("precip"), new=c("mean_precip"))
precip_dt <- within(precip_dt, remove("ClimateScenario"))

### Read gdd
gdd_dt <- data.table(readRDS(paste0(main_in_dir, "gdd_usa.rds")))

#
# merge several data frames:
#
all_data_dt <- Reduce(function(...) merge(..., all = T), 
                      list(FF_dt, gen_dt, diapause_dt, precip_dt, gdd_dt))

# there are NA values in escaped population of 4th generations!
# I have to look into it to see why, but perhaps the reason is
# that there is no such a thing. So, for now, I replace them with zeros

all_data_dt$mean_escaped_Gen4[is.na(all_data_dt$mean_escaped_Gen4)] <- 0
all_data_dt$mean_escaped_Gen3[is.na(all_data_dt$mean_escaped_Gen3)] <- 0
all_data_dt$treatment <- 0
all_data_dt$ClimateScenario = "observed"

saveRDS(all_data_dt, paste0(main_out_dir, "all_data_usa.rds"))

write.csv(all_data_dt, 
          paste0(main_out_dir, "all_data_usa.csv"),
          row.names=FALSE)



